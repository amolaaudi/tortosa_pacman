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


# Pathfinding helpers
def bfs_next_action(game_state, start, goals, blocked=None):
    if not goals:
        return None
    goal_set = set(goals)
    blocked_set = set(blocked) if blocked else set()

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
                if npos not in visited and npos not in blocked_set:
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


def count_exits(game_state, pos):
    walls = game_state.get_walls()
    x, y = int(pos[0]), int(pos[1])
    n = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
            n += 1
    return n


def ghost_danger_zone(ghost_positions, radius=2):
    zone = set()
    for gx, gy in ghost_positions:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) <= radius:
                    zone.add((int(gx) + dx, int(gy) + dy))
    return zone


def build_tunnel_map(game_state):
    """
    For each open tile, compute how far it is from the nearest junction
    (a tile with 3+ open neighbors). High value = deep in a corridor.
    """
    walls = game_state.get_walls()
    w, h = walls.width, walls.height

    def num_exits(x, y):
        c = 0
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not walls[nx][ny]:
                c += 1
        return c

    open_tiles = set()
    junctions = set()
    for x in range(w):
        for y in range(h):
            if not walls[x][y]:
                open_tiles.add((x, y))
                if num_exits(x, y) >= 3:
                    junctions.add((x, y))

    # multi-source BFS from all junctions
    dist = {pos: 999 for pos in open_tiles}
    queue = util.Queue()
    for j in junctions:
        dist[j] = 0
        queue.push(j)

    while not queue.is_empty():
        pos = queue.pop()
        x, y = pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            npos = (nx, ny)
            if npos in open_tiles and dist[npos] == 999:
                dist[npos] = dist[pos] + 1
                queue.push(npos)

    return dist


# Particle filter to track opponents we can't see
class ParticleFilter:
    def __init__(self, num_particles=300):
        self.num_particles = num_particles
        self.particles = []

    def initialize(self, game_state, our_red):
        walls = game_state.get_walls()
        w, h = walls.width, walls.height
        mid = w // 2
        # opponents start on the opposite half
        if our_red:
            xs = range(mid, w)
        else:
            xs = range(0, mid)
        candidates = [(x, y) for x in xs for y in range(h) if not walls[x][y]]
        if not candidates:
            candidates = [(x, y) for x in range(w) for y in range(h) if not walls[x][y]]
        self.particles = [random.choice(candidates) for _ in range(self.num_particles)]

    def predict(self, game_state):
        walls = game_state.get_walls()
        w, h = walls.width, walls.height
        new_p = []
        for px, py in self.particles:
            ix, iy = int(px), int(py)
            neighbors = [(ix, iy)]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < w and 0 <= ny < h and not walls[nx][ny]:
                    neighbors.append((nx, ny))
            new_p.append(random.choice(neighbors))
        self.particles = new_p

    def observe(self, my_pos, noisy_dist, game_state):
        """Reweight particles based on the noisy distance reading we got."""
        if noisy_dist is None:
            return
        # noisy distance has error in [-6, 6], so weight particles by how
        # well they match the observed distance
        weights = []
        for p in self.particles:
            diff = abs(util.manhattan_distance(my_pos, p) - noisy_dist)
            weights.append(max(0.01, 7 - diff) if diff <= 6 else 0.01)

        total = sum(weights)
        if total < 0.1:
            self.initialize(game_state, True)
            return

        # resample proportional to weights
        new_p = []
        for _ in range(self.num_particles):
            r = random.uniform(0, total)
            acc = 0
            for i, wv in enumerate(weights):
                acc += wv
                if acc >= r:
                    new_p.append(self.particles[i])
                    break
            else:
                new_p.append(self.particles[-1])
        self.particles = new_p

    def reset_to(self, pos):
        ipos = (int(pos[0]), int(pos[1]))
        self.particles = [ipos] * self.num_particles

    def get_estimate(self):
        if not self.particles:
            return None
        counts = {}
        for p in self.particles:
            counts[p] = counts.get(p, 0) + 1
        return max(counts, key=counts.get)


# Offensive agent
class OffensiveAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.border = []
        self.tunnel_depth = {}

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.border = boundary_positions(game_state, self.red)
        self.tunnel_depth = build_tunnel_map(game_state)

    def _home_dist(self, pos):
        if not self.border:
            return 0
        return min(self.get_maze_distance(pos, b) for b in self.border)

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        actions = game_state.get_legal_actions(self.index)
        carrying = my_state.num_carrying
        food_left = len(self.get_food(game_state).as_list())
        score = self.get_score(game_state)
        time_left = game_state.data.timeleft
        home_d = self._home_dist(my_pos)

        opps = [game_state.get_agent_state(o) for o in self.get_opponents(game_state)]
        active_ghosts = [s for s in opps
                         if s.get_position() is not None
                         and not s.is_pacman
                         and s.scared_timer <= 2]
        scared_ghosts = [s for s in opps
                         if s.get_position() is not None
                         and not s.is_pacman
                         and s.scared_timer > 2]
        ghost_pos = [s.get_position() for s in active_ghosts]

        if food_left <= 2:
            return self._go_home(game_state, my_pos, actions)

        if carrying > 0 and time_left < home_d * 2 + 25:
            return self._go_home(game_state, my_pos, actions)

        # handle nearby ghosts
        if my_state.is_pacman and ghost_pos:
            dists = {g: self.get_maze_distance(my_pos, g) for g in ghost_pos}
            closest_d = min(dists.values())
            close = [g for g, d in dists.items() if d <= 6]

            if close:
                cap_a = self._capsule_escape(game_state, my_pos, close)
                if cap_a is not None:
                    return cap_a

                if count_exits(game_state, my_pos) == 1 and closest_d <= 3:
                    return self._flee(game_state, my_pos, close, actions)

                if carrying >= 2 or closest_d <= 4:
                    safe_a = self._safe_home(game_state, my_pos, actions, ghost_pos)
                    return safe_a if safe_a else self._flee(game_state, my_pos, close, actions)

                return self._flee(game_state, my_pos, close, actions)

        if carrying > 0 and home_d <= 2:
            return self._go_home(game_state, my_pos, actions)

        carry_limit = self._carry_limit(my_pos, score, bool(ghost_pos))
        if carrying >= carry_limit:
            safe_a = self._safe_home(game_state, my_pos, actions, ghost_pos)
            return safe_a if safe_a else self._go_home(game_state, my_pos, actions)

        if scared_ghosts:
            target = min(scared_ghosts,
                         key=lambda s: self.get_maze_distance(my_pos, s.get_position()))
            a = bfs_next_action(game_state, my_pos, [target.get_position()])
            if a is not None:
                return a

        # if a lot of food is blocked by ghosts, try eating a capsule first
        cap_a = self._proactive_capsule(game_state, my_pos, ghost_pos)
        if cap_a is not None:
            return cap_a

        food_list = self.get_food(game_state).as_list()
        target = self._pick_food(my_pos, food_list, ghost_pos)
        if target is not None:
            blocked = ghost_danger_zone(ghost_pos, radius=2) if ghost_pos else None
            a = bfs_next_action(game_state, my_pos, [target], blocked)
            if a is None:
                a = bfs_next_action(game_state, my_pos, [target])
            if a is not None:
                return a

        non_stop = [a for a in actions if a != Directions.STOP]
        return random.choice(non_stop if non_stop else actions)

    def _carry_limit(self, my_pos, score, ghosts_present):
        home_d = self._home_dist(my_pos)
        if score <= -6:
            base = 8
        elif score >= 6:
            base = 2
        else:
            base = 4

        if ghosts_present:
            base = max(2, base - 2)

        if home_d <= 3:
            base = max(base, 7)

        return base

    def _proactive_capsule(self, game_state, my_pos, ghost_pos):
        if not ghost_pos:
            return None
        capsules = self.get_capsules(game_state)
        if not capsules:
            return None
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return None

        # count food that's effectively blocked by ghosts
        blocked_food = [f for f in food_list
                        if any(self.get_maze_distance(f, g) <= 4 for g in ghost_pos)]
        if len(blocked_food) < 5:
            return None

        best_cap = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
        d_cap = self.get_maze_distance(my_pos, best_cap)
        d_food = min(self.get_maze_distance(my_pos, f) for f in food_list)

        # only worth it if the capsule is roughly on the way
        if d_cap <= d_food + 2:
            a = bfs_next_action(game_state, my_pos, [best_cap])
            if a is not None:
                return a
        return None

    def _capsule_escape(self, game_state, my_pos, ghosts):
        for cap in self.get_capsules(game_state):
            d_me = self.get_maze_distance(my_pos, cap)
            d_ghost = min(self.get_maze_distance(g, cap) for g in ghosts)
            if d_me < d_ghost:
                a = bfs_next_action(game_state, my_pos, [cap])
                if a is not None:
                    return a
        return None

    def _flee(self, game_state, my_pos, ghosts, actions):
        best_a = Directions.STOP
        best_s = -float('inf')
        for action in actions:
            if action == Directions.STOP:
                continue
            succ = game_state.generate_successor(self.index, action)
            pos = nearest_point(succ.get_agent_state(self.index).get_position())
            g_dist = min(self.get_maze_distance(pos, g) for g in ghosts)
            h_dist = self._home_dist(pos)
            exits = count_exits(game_state, pos)
            depth = self.tunnel_depth.get(pos, 0)
            s = 3 * g_dist - h_dist + (0 if exits > 1 else -10) - depth * 2
            if s > best_s:
                best_s = s
                best_a = action
        return best_a

    def _safe_home(self, game_state, my_pos, actions, ghost_pos):
        if not ghost_pos:
            return self._go_home(game_state, my_pos, actions)
        blocked = ghost_danger_zone(ghost_pos, radius=2)
        return bfs_next_action(game_state, my_pos, self.border, blocked)

    def _go_home(self, game_state, my_pos, actions):
        a = bfs_next_action(game_state, my_pos, self.border)
        if a is not None:
            return a
        non_stop = [a for a in actions if a != Directions.STOP]
        return random.choice(non_stop if non_stop else actions)

    def _pick_food(self, my_pos, food_list, ghost_pos):
        if not food_list:
            return None
        if not ghost_pos:
            def cluster_val(f):
                neighbors = sum(1 for f2 in food_list
                                if f != f2 and abs(f[0]-f2[0]) + abs(f[1]-f2[1]) <= 4)
                return self.get_maze_distance(my_pos, f) - neighbors * 2
            return min(food_list, key=cluster_val)

        min_g = min(self.get_maze_distance(my_pos, g) for g in ghost_pos)

        def safety_val(f):
            dist = self.get_maze_distance(my_pos, f)
            ghost_d = min(self.get_maze_distance(f, g) for g in ghost_pos)
            # avoid food deep in tunnels when ghost is close
            depth_penalty = self.tunnel_depth.get(f, 0) if min_g <= 6 else 0
            return dist - ghost_d + depth_penalty
        return min(food_list, key=safety_val)


# Defensive agent
class DefensiveAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.border = []
        self._patrol = []
        self._pidx = 0
        self._prev_food = None
        self._filters = {}  # particle filter per opponent

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.border = boundary_positions(game_state, self.red)
        self._patrol = self._build_patrol(game_state)
        self._prev_food = set(self.get_food_you_are_defending(game_state).as_list())

        for opp_idx in self.get_opponents(game_state):
            pf = ParticleFilter(num_particles=300)
            pf.initialize(game_state, self.red)
            self._filters[opp_idx] = pf

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        actions = game_state.get_legal_actions(self.index)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        cur_food = set(self.get_food_you_are_defending(game_state).as_list())
        eaten    = self._prev_food - cur_food
        self._prev_food = cur_food

        # update particle filters
        noisy = game_state.get_agent_distances()
        for opp_idx in self.get_opponents(game_state):
            state = game_state.get_agent_state(opp_idx)
            pf = self._filters[opp_idx]
            if state.get_position() is not None:
                pf.reset_to(state.get_position())
            else:
                pf.predict(game_state)
                if noisy[opp_idx] is not None:
                    pf.observe(my_pos, noisy[opp_idx], game_state)

        # chase visible invader
        if invaders:
            nearest = min(invaders, key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
            target  = nearest.get_position()

            if my_state.scared_timer > 0:
                return self._shadow(game_state, target, actions)

            a = bfs_next_action(game_state, my_pos, [target])
            if a is not None:
                return a

        # go to last tile that was eaten (fresh trail)
        if eaten:
            target = min(eaten, key=lambda p: self.get_maze_distance(my_pos, p))
            a = bfs_next_action(game_state, my_pos, [target])
            if a is not None:
                return a

        # use particle filter to guess where hidden invaders are
        guess = self._pf_guess(game_state)
        if guess is not None:
            a = bfs_next_action(game_state, my_pos, [guess])
            if a is not None:
                return a

        return self._do_patrol(game_state, my_pos, actions)

    def _pf_guess(self, game_state):
        """Use particle filter to find the most likely position of a hidden invader."""
        layout = game_state.data.layout
        mid = layout.width // 2
        walls = game_state.get_walls()

        best_pos = None
        best_count = 0

        for opp_idx in self.get_opponents(game_state):
            state = game_state.get_agent_state(opp_idx)
            if state.get_position() is not None:
                continue
            if not state.is_pacman:
                continue
            pf = self._filters[opp_idx]
            estimate = pf.get_estimate()
            if estimate is None:
                continue
            ex, ey = estimate
            on_our_side = (ex < mid) if self.red else (ex >= mid)
            if on_our_side and not walls[ex][ey]:
                count = pf.particles.count(estimate)
                if count > best_count:
                    best_count = count
                    best_pos   = estimate

        return best_pos

    def _shadow(self, game_state, threat, actions):
        # stay ~3 tiles away so we can re-engage right when the timer expires
        best_a = Directions.STOP
        best_s = -float('inf')
        for action in actions:
            if action == Directions.STOP:
                continue
            succ = game_state.generate_successor(self.index, action)
            pos = nearest_point(succ.get_agent_state(self.index).get_position())
            d = self.get_maze_distance(pos, threat)
            s = -abs(d - 3)
            if s > best_s:
                best_s = s
                best_a = action
        return best_a

    def _do_patrol(self, game_state, my_pos, actions):
        if not self._patrol:
            non_stop = [a for a in actions if a != Directions.STOP]
            return random.choice(non_stop if non_stop else actions)

        target = self._patrol[self._pidx % len(self._patrol)]
        if self.get_maze_distance(my_pos, target) <= 1:
            self._pidx = (self._pidx + 1) % len(self._patrol)
            target = self._patrol[self._pidx]

        a = bfs_next_action(game_state, my_pos, [target])
        if a is not None:
            return a

        non_stop = [a for a in actions if a != Directions.STOP]
        return random.choice(non_stop if non_stop else actions)

    def _build_patrol(self, game_state):
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            step = max(1, len(self.border) // 5)
            return self.border[::step][:5]

        def food_cover(b):
            return min(util.manhattan_distance(b, f) for f in food)

        sorted_border = sorted(self.border, key=food_cover)

        # pick spread out tiles so we cover different parts of the border
        selected = []
        used_buckets = set()
        for b in sorted_border:
            bucket = b[1] // 4
            if bucket not in used_buckets:
                selected.append(b)
                used_buckets.add(bucket)
            if len(selected) >= 5:
                break

        return selected if selected else self.border[:5]
