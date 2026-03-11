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


# BFS returning the first action needed to reach any of the goal positions
def bfs_next_action(game_state, start, goals):
    if not goals:
        return None
    goal_set = set(goals)
    if start in goal_set:
        return None

    walls = game_state.get_walls()
    w, h = walls.width, walls.height

    queue = util.Queue()
    queue.push((start, None))
    visited = {start}

    dirs = [
        (Directions.NORTH, (0,  1)),
        (Directions.SOUTH, (0, -1)),
        (Directions.EAST,  (1,  0)),
        (Directions.WEST,  (-1, 0)),
    ]

    while not queue.is_empty():
        pos, first_move = queue.pop()
        x, y = int(pos[0]), int(pos[1])
        for action, (dx, dy) in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not walls[nx][ny]:
                npos = (nx, ny)
                if npos not in visited:
                    move = first_move if first_move is not None else action
                    if npos in goal_set:
                        return move
                    visited.add(npos)
                    queue.push((npos, move))
    return None


def boundary_positions(game_state, red_team):
    walls = game_state.get_walls()
    mid = game_state.data.layout.width // 2
    x = (mid - 1) if red_team else mid
    return [(x, y) for y in range(game_state.data.layout.height) if not walls[x][y]]


def dist_to_boundary(agent, game_state):
    pos = game_state.get_agent_state(agent.index).get_position()
    border = boundary_positions(game_state, agent.red)
    if not border:
        return 0
    return min(agent.get_maze_distance(pos, b) for b in border)


def count_exits(game_state, pos):
    """how many directions can we move from this tile (used to detect dead ends)"""
    walls = game_state.get_walls()
    x, y = int(pos[0]), int(pos[1])
    count = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
            count += 1
    return count


class OffensiveAgent(CaptureAgent):
    """Collects food on enemy side and returns to score."""

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start  = None
        self.border = []

    def register_initial_state(self, game_state):
        self.start  = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.border = boundary_positions(game_state, self.red)  # walls never change

    def _home_dist(self, my_pos):
        if not self.border:
            return 0
        return min(self.get_maze_distance(my_pos, b) for b in self.border)

    def _carry_limit(self, my_pos):
        # carry more food when we're closer to home — less risk
        d = self._home_dist(my_pos)
        if d <= 3:
            return 6
        if d <= 7:
            return 4
        return 2

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos   = my_state.get_position()
        actions  = game_state.get_legal_actions(self.index)
        carrying = my_state.num_carrying
        food_left = len(self.get_food(game_state).as_list())

        opps = [game_state.get_agent_state(o) for o in self.get_opponents(game_state)]
        active_ghosts = [s.get_position() for s in opps
                         if s.get_position() is not None and not s.is_pacman and s.scared_timer <= 0]
        scared_ghosts = [s for s in opps
                         if s.get_position() is not None and not s.is_pacman and s.scared_timer > 0]

        # with little food left or almost out of time just go home
        time_left = game_state.data.timeleft
        if food_left <= 2 or (carrying > 0 and time_left < self._home_dist(my_pos) + 20):
            return self._go_home(game_state, my_pos, actions)

        # ghost is dangerously close
        if my_state.is_pacman and active_ghosts:
            dists = {g: self.get_maze_distance(my_pos, g) for g in active_ghosts}
            closest_dist = min(dists.values())
            danger_thresh = 5 + min(carrying, 3)

            if closest_dist <= danger_thresh:
                close = [g for g, d in dists.items() if d <= danger_thresh]

                # try to grab a capsule if we can get there first
                cap_a = self._capsule_escape(game_state, my_pos, close)
                if cap_a is not None:
                    return cap_a

                # if we're in a dead end and ghost is very close, head home
                if count_exits(game_state, my_pos) == 1 and closest_dist <= 3:
                    return self._go_home(game_state, my_pos, actions)

                return self._flee(game_state, my_pos, close, actions)

        # close to border with goods — bank them
        if carrying > 0 and self._home_dist(my_pos) <= 2:
            return self._go_home(game_state, my_pos, actions)

        # hit carry limit
        if carrying >= self._carry_limit(my_pos):
            return self._go_home(game_state, my_pos, actions)

        # ghost is scared — chase it for bonus points
        if scared_ghosts:
            target = min(scared_ghosts,
                         key=lambda s: self.get_maze_distance(my_pos, s.get_position()))
            a = bfs_next_action(game_state, my_pos, [target.get_position()])
            if a is not None:
                return a

        # go eat food
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
            d_me    = self.get_maze_distance(my_pos, cap)
            d_ghost = min(self.get_maze_distance(g, cap) for g in ghosts)
            if d_me < d_ghost:
                a = bfs_next_action(game_state, my_pos, [cap])
                if a is not None:
                    return a
        return None

    def _flee(self, game_state, my_pos, ghosts, actions):
        best_action = Directions.STOP
        best_score  = -float('inf')

        for action in actions:
            if action == Directions.STOP:
                continue
            succ = game_state.generate_successor(self.index, action)
            pos  = succ.get_agent_state(self.index).get_position()
            pos  = nearest_point(pos)

            ghost_dist = min(self.get_maze_distance(pos, g) for g in ghosts)
            home_dist  = min(self.get_maze_distance(pos, b) for b in self.border) if self.border else 0
            exits      = count_exits(game_state, pos)

            # prefer moving away from ghosts and toward home, avoid dead ends
            score = 2 * ghost_dist - home_dist + (0 if exits > 1 else -5)
            if score > best_score:
                best_score  = score
                best_action = action

        return best_action

    def _go_home(self, game_state, my_pos, actions):
        a = bfs_next_action(game_state, my_pos, self.border)
        if a is not None:
            return a
        non_stop = [a for a in actions if a != Directions.STOP]
        return random.choice(non_stop if non_stop else actions)

    def _pick_food(self, game_state, my_pos, food_list, ghosts):
        if not food_list:
            return None
        if not ghosts:
            return min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        # balance closeness to us vs distance from ghosts
        return min(food_list,
                   key=lambda f: self.get_maze_distance(my_pos, f)
                               - min(self.get_maze_distance(f, g) for g in ghosts))


class DefensiveAgent(CaptureAgent):
    """Guards home side. Chases visible invaders, patrols when none visible."""

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start  = None
        self._posts = []
        self._post_idx = 0

    def register_initial_state(self, game_state):
        self.start  = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self._posts = self._compute_patrol_posts(game_state)

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos   = my_state.get_position()
        actions  = game_state.get_legal_actions(self.index)

        enemies  = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        if invaders:
            nearest = min(invaders,
                          key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
            target  = nearest.get_position()

            if my_state.scared_timer > 0:
                return self._back_off(game_state, my_pos, target, actions)

            a = bfs_next_action(game_state, my_pos, [target])
            if a is not None:
                return a

        # no visible invader — use noisy distances to guess where enemies might be
        noisy = game_state.get_agent_distances()
        suspect = self._guess_invader_pos(game_state, enemies, noisy)
        if suspect is not None:
            a = bfs_next_action(game_state, my_pos, [suspect])
            if a is not None:
                return a

        return self._patrol(game_state, my_pos, actions)

    def _guess_invader_pos(self, game_state, enemies, noisy):
        """Pick a patrol tile near where a noisy signal suggests an enemy is."""
        if noisy is None:
            return None
        walls  = game_state.get_walls()
        layout = game_state.data.layout
        mid    = layout.width // 2

        for e in enemies:
            idx   = e.get_position()  # None when not visible
            if idx is not None:
                continue
            # grab the noisy distance for this opponent
            opp_idx = [i for i in self.get_opponents(game_state)
                       if game_state.get_agent_state(i) is e]
            if not opp_idx:
                continue
            noisy_d = noisy[opp_idx[0]]
            if noisy_d is None:
                continue
            # look for reachable tiles in our half at roughly that distance
            my_pos = game_state.get_agent_state(self.index).get_position()
            x_range = range(1, mid) if self.red else range(mid, layout.width - 1)
            candidates = []
            for x in x_range:
                for y in range(1, layout.height - 1):
                    if not walls[x][y]:
                        d = util.manhattan_distance(my_pos, (x, y))
                        if abs(d - noisy_d) <= 3:
                            candidates.append((x, y))
            if candidates:
                return random.choice(candidates)
        return None

    def _back_off(self, game_state, my_pos, threat, actions):
        best_action = Directions.STOP
        best_dist   = -1
        for action in actions:
            if action == Directions.STOP:
                continue
            succ = game_state.generate_successor(self.index, action)
            pos  = nearest_point(succ.get_agent_state(self.index).get_position())
            d    = self.get_maze_distance(pos, threat)
            if d > best_dist:
                best_dist   = d
                best_action = action
        return best_action

    def _patrol(self, game_state, my_pos, actions):
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
        walls  = game_state.get_walls()
        layout = game_state.data.layout
        mid    = layout.width // 2

        x_range = range(max(1, mid - 4), mid) if self.red else range(mid, min(layout.width - 1, mid + 4))
        candidates = [(x, y) for x in x_range
                      for y in range(1, layout.height - 1) if not walls[x][y]]

        if not candidates:
            return [self.start]

        if not food_to_defend:
            step = max(1, len(candidates) // 6)
            return candidates[::step][:6]

        candidates.sort(key=lambda p: min(util.manhattan_distance(p, f) for f in food_to_defend))

        selected  = []
        used_rows = set()
        for pos in candidates:
            bucket = pos[1] // 3
            if bucket not in used_rows:
                selected.append(pos)
                used_rows.add(bucket)
            if len(selected) >= 6:
                break

        return selected if selected else candidates[:6]
