# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


# BFS to find the first action to take to reach the closest goal
def bfs_next_action(game_state, start, goals):
    if not goals:
        return None
    goal_set = set(goals)
    if start in goal_set:
        return None

    walls = game_state.get_walls()
    w, h = walls.width, walls.height

    frontier = util.Queue()
    frontier.push((start, None))
    visited = {start}

    step_dirs = [
        (Directions.NORTH, (0, 1)),
        (Directions.SOUTH, (0, -1)),
        (Directions.EAST,  (1, 0)),
        (Directions.WEST,  (-1, 0)),
    ]

    while not frontier.is_empty():
        pos, first_action = frontier.pop()
        x, y = int(pos[0]), int(pos[1])

        for action, (dx, dy) in step_dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not walls[nx][ny]:
                npos = (nx, ny)
                if npos not in visited:
                    taken = first_action if first_action is not None else action
                    if npos in goal_set:
                        return taken
                    visited.add(npos)
                    frontier.push((npos, taken))

    return None


def boundary_positions(game_state, red_team):
    layout = game_state.data.layout
    walls = game_state.get_walls()
    mid = layout.width // 2
    x = (mid - 1) if red_team else mid
    return [(x, y) for y in range(layout.height) if not walls[x][y]]


def dist_to_boundary(agent, game_state):
    pos = game_state.get_agent_state(agent.index).get_position()
    border = boundary_positions(game_state, agent.red)
    if not border:
        return 0
    return min(agent.get_maze_distance(pos, b) for b in border)


class OffensiveAgent(CaptureAgent):
    """Goes into enemy territory to collect food and comes back to score."""

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.carry_limit = 3

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        actions = game_state.get_legal_actions(self.index)
        food_left = len(self.get_food(game_state).as_list())
        carrying = my_state.num_carrying

        opps = [game_state.get_agent_state(o) for o in self.get_opponents(game_state)]
        active_ghosts = [s.get_position() for s in opps
                         if s.get_position() is not None and not s.is_pacman and s.scared_timer <= 0]
        scared_ghosts = [s for s in opps
                         if s.get_position() is not None and not s.is_pacman and s.scared_timer > 0]

        # only 2 food left, just go home
        if food_left <= 2:
            return self._go_home(game_state, my_pos, actions)

        # ghost is too close while we are pacman
        if my_state.is_pacman and active_ghosts:
            dists = {g: self.get_maze_distance(my_pos, g) for g in active_ghosts}
            closest = min(dists.values())

            danger = 4 + min(carrying, 3)

            if closest <= danger:
                close = [g for g, d in dists.items() if d <= danger]
                cap_a = self._capsule_escape(game_state, my_pos, close)
                if cap_a is not None:
                    return cap_a
                return self._flee_action(game_state, my_pos, close, actions)

        # almost at the border with food, bank it
        if carrying > 0 and dist_to_boundary(self, game_state) <= 2:
            return self._go_home(game_state, my_pos, actions)

        # carrying enough, go back
        if carrying >= self.carry_limit:
            return self._go_home(game_state, my_pos, actions)

        # scared ghost nearby, eat it
        if scared_ghosts:
            nearest_sg = min(scared_ghosts,
                             key=lambda s: self.get_maze_distance(my_pos, s.get_position()))
            a = bfs_next_action(game_state, my_pos, [nearest_sg.get_position()])
            if a is not None:
                return a

        # go get food
        food_list = self.get_food(game_state).as_list()
        target = self._pick_food(game_state, my_pos, food_list, active_ghosts)
        if target is not None:
            a = bfs_next_action(game_state, my_pos, [target])
            if a is not None:
                return a

        non_stop = [a for a in actions if a != Directions.STOP]
        return random.choice(non_stop if non_stop else actions)

    def _capsule_escape(self, game_state, my_pos, ghosts):
        for cap in self.get_capsules(game_state):
            dist_me = self.get_maze_distance(my_pos, cap)
            dist_ghost = min(self.get_maze_distance(g, cap) for g in ghosts)
            if dist_me < dist_ghost:
                a = bfs_next_action(game_state, my_pos, [cap])
                if a is not None:
                    return a
        return None

    def _flee_action(self, game_state, my_pos, ghosts, actions):
        # try to run away from ghosts and head home
        border = boundary_positions(game_state, self.red)
        best_action = Directions.STOP
        best_score = -float('inf')

        for action in actions:
            if action == Directions.STOP:
                continue
            succ = game_state.generate_successor(self.index, action)
            pos = succ.get_agent_state(self.index).get_position()
            if pos != nearest_point(pos):
                pos = nearest_point(pos)

            min_ghost = min(self.get_maze_distance(pos, g) for g in ghosts)
            home_d = (min(self.get_maze_distance(pos, b) for b in border)
                      if border else 0)
            score = 2 * min_ghost - home_d

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _go_home(self, game_state, my_pos, actions):
        border = boundary_positions(game_state, self.red)
        a = bfs_next_action(game_state, my_pos, border)
        if a is not None:
            return a
        non_stop = [a for a in actions if a != Directions.STOP]
        return random.choice(non_stop if non_stop else actions)

    def _pick_food(self, game_state, my_pos, food_list, ghosts):
        # pick the safest food to go for
        if not food_list:
            return None
        if not ghosts:
            return min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        return min(food_list, key=lambda f: self.get_maze_distance(my_pos, f) - min(self.get_maze_distance(f, g) for g in ghosts))


class DefensiveAgent(CaptureAgent):
    """Stays on our side and chases invaders. Patrols near the middle when idle."""

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self._posts = []
        self._post_idx = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self._posts = self._compute_patrol_posts(game_state)

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        actions = game_state.get_legal_actions(self.index)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        if invaders:
            nearest = min(invaders,
                          key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
            target = nearest.get_position()

            if my_state.scared_timer > 0:
                return self._retreat_from(game_state, my_pos, target, actions)

            a = bfs_next_action(game_state, my_pos, [target])
            if a is not None:
                return a

        return self._patrol_action(game_state, my_pos, actions)

    def _retreat_from(self, game_state, my_pos, threat, actions):
        best_action = Directions.STOP
        best_dist = -1
        for action in actions:
            if action == Directions.STOP:
                continue
            succ = game_state.generate_successor(self.index, action)
            pos = succ.get_agent_state(self.index).get_position()
            if pos != nearest_point(pos):
                pos = nearest_point(pos)
            d = self.get_maze_distance(pos, threat)
            if d > best_dist:
                best_dist = d
                best_action = action
        return best_action

    def _patrol_action(self, game_state, my_pos, actions):
        if not self._posts:
            non_stop = [a for a in actions if a != Directions.STOP]
            return random.choice(non_stop if non_stop else actions)

        target = self._posts[self._post_idx % len(self._posts)]

        if self.get_maze_distance(my_pos, target) <= 1:
            self._post_idx = (self._post_idx + 1) % len(self._posts)
            target = self._posts[self._post_idx]

        a = bfs_next_action(game_state, my_pos, [target])
        if a is not None:
            return a

        non_stop = [a for a in actions if a != Directions.STOP]
        return random.choice(non_stop if non_stop else actions)

    def _compute_patrol_posts(self, game_state):
        food_to_defend = self.get_food_you_are_defending(game_state).as_list()
        walls = game_state.get_walls()
        layout = game_state.data.layout
        mid = layout.width // 2

        cols = range(max(1, mid - 4), mid) if self.red else range(mid, min(layout.width - 1, mid + 4))

        candidates = []
        for x in cols:
            for y in range(1, layout.height - 1):
                if not walls[x][y]:
                    candidates.append((x, y))

        if not candidates:
            return [self.start]

        if not food_to_defend:
            step = max(1, len(candidates) // 6)
            return candidates[::step][:6]

        candidates.sort(key=lambda pos: min(util.manhattan_distance(pos, f) for f in food_to_defend))

        # pick up to 6 spots spread across different heights
        selected = []
        used_ys = set()
        for pos in candidates:
            y_bucket = pos[1] // 3
            if y_bucket not in used_ys:
                selected.append(pos)
                used_ys.add(y_bucket)
            if len(selected) >= 6:
                break

        return selected if selected else candidates[:6]
