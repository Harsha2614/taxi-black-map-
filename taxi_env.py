"""
Taxi-v3-style environment with extended traffic and weather metadata.
"""
import numpy as np
import random

# Fixed pickup/dropoff locations across the grid.
GRID_SIZE = 6
LOCS = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (0, 5), (1, 0), (1, 2), (1, 3), (1, 5),
    (2, 1), (2, 2), (2, 4), (2, 5), (3, 0),
    (3, 1), (3, 3), (3, 4), (4, 0), (4, 2),
    (4, 3), (4, 5), (5, 1), (5, 3), (5, 5),
]
LOC_LABELS = [
    'R', 'G', 'Y', 'B', 'P', 'O', 'J', 'S', 'M', 'A',
    'K', 'T', 'C', 'E', 'D', 'L', 'N', 'Q', 'U', 'V',
    'W', 'X', 'Z', 'H', 'I',
]
LOC_COLORS = [
    '#ef4444', '#10b981', '#f5c842', '#3b82f6', '#f97316',
    '#a855f7', '#14b8a6', '#ec4899', '#84cc16', '#06b6d4',
    '#fb7185', '#38bdf8', '#22c55e', '#f59e0b', '#c084fc',
    '#f43f5e', '#2dd4bf', '#818cf8', '#bef264', '#facc15',
    '#fb923c', '#67e8f9', '#d946ef', '#4ade80', '#93c5fd',
]

# Walls in the grid (kept for future expansion).
WALLS = {
    (0, 1, 'S'), (1, 1, 'S'),
    (3, 0, 'S'), (4, 0, 'S'),
    (3, 2, 'S'), (4, 2, 'S'),
}

ACTION_NAMES = ['South', 'North', 'East', 'West', 'Pickup', 'Drop-off']
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

NUM_LOCATIONS = len(LOCS)
PASSENGER_STATES = NUM_LOCATIONS + 1
NUM_STATES = (GRID_SIZE * GRID_SIZE) * PASSENGER_STATES * NUM_LOCATIONS
NUM_ACTIONS = 6


def encode_state(taxi_row, taxi_col, pass_loc, destination):
    """Encode (row, col, passenger, destination) into a dense integer state."""
    i = taxi_row
    i = i * GRID_SIZE + taxi_col
    i = i * PASSENGER_STATES + pass_loc
    i = i * NUM_LOCATIONS + destination
    return i


def decode_state(state):
    """Decode an integer state into (taxi_row, taxi_col, pass_loc, destination)."""
    destination = state % NUM_LOCATIONS
    state //= NUM_LOCATIONS
    pass_loc = state % PASSENGER_STATES
    state //= PASSENGER_STATES
    taxi_col = state % GRID_SIZE
    state //= GRID_SIZE
    taxi_row = state
    return taxi_row, taxi_col, pass_loc, destination


class TaxiEnv:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.taxi_row = self.rng.randint(0, GRID_SIZE - 1)
        self.taxi_col = self.rng.randint(0, GRID_SIZE - 1)
        self.pass_loc = self.rng.randint(0, NUM_LOCATIONS - 1)
        self.destination = self.rng.randint(0, NUM_LOCATIONS - 1)
        while self.destination == self.pass_loc:
            self.destination = self.rng.randint(0, NUM_LOCATIONS - 1)
        self.pass_on_board = False
        self.done = False
        self.steps = 0
        self.total_reward = 0.0
        self.pickup_time = None
        self.dropoff_time = None
        return encode_state(self.taxi_row, self.taxi_col, self.pass_loc, self.destination)

    def can_move(self, row, col, action):
        """Check whether a move stays inside the grid."""
        dr, dc = MOVES[action]
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
            return False, nr, nc
        return True, nr, nc

    def step(self, action):
        if self.done:
            return encode_state(
                self.taxi_row,
                self.taxi_col,
                NUM_LOCATIONS if self.pass_on_board else self.pass_loc,
                self.destination,
            ), 0, True, {}

        reward = -1
        info = {'action_name': ACTION_NAMES[action], 'event': 'move'}

        if action < 4:
            ok, nr, nc = self.can_move(self.taxi_row, self.taxi_col, action)
            if ok:
                self.taxi_row, self.taxi_col = nr, nc
            else:
                info['event'] = 'wall'

        elif action == 4:
            pick_loc = LOCS[self.pass_loc]
            if (
                not self.pass_on_board
                and self.taxi_row == pick_loc[0]
                and self.taxi_col == pick_loc[1]
            ):
                self.pass_on_board = True
                self.pickup_time = self.steps
                info['event'] = 'pickup'
            else:
                reward = -10
                info['event'] = 'illegal_pickup'

        elif action == 5:
            dest_loc = LOCS[self.destination]
            if (
                self.pass_on_board
                and self.taxi_row == dest_loc[0]
                and self.taxi_col == dest_loc[1]
            ):
                self.pass_on_board = False
                self.dropoff_time = self.steps
                reward = 20
                self.done = True
                info['event'] = 'dropoff'
            else:
                reward = -10
                info['event'] = 'illegal_dropoff'

        self.steps += 1
        self.total_reward += reward

        pass_state = NUM_LOCATIONS if self.pass_on_board else self.pass_loc
        next_state = encode_state(self.taxi_row, self.taxi_col, pass_state, self.destination)

        info.update({
            'taxi_pos': (self.taxi_row, self.taxi_col),
            'pass_loc': self.pass_loc,
            'destination': self.destination,
            'pass_on_board': self.pass_on_board,
            'pass_label': LOC_LABELS[self.pass_loc],
            'dest_label': LOC_LABELS[self.destination],
            'reward': reward,
            'total_reward': self.total_reward,
            'steps': self.steps,
        })
        return next_state, reward, self.done, info


class TaxiEnvExtended(TaxiEnv):
    """Extended version with traffic, weather, energy, and satisfaction."""

    def __init__(self, seed=None):
        self.traffic_grid = np.zeros((5, 5))
        self.weather_grid = np.zeros((5, 5))
        self.energy = 100.0
        self.sat_bonus = 0.0
        super().__init__(seed)
        self._generate_hazards(seed)

    def _generate_hazards(self, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else self.np_rng
        self.traffic_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.weather_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for _ in range(6):
            r, c = rng.randint(0, GRID_SIZE, 2)
            self.traffic_grid[r][c] = rng.uniform(0.3, 1.0)
        for _ in range(5):
            r, c = rng.randint(0, GRID_SIZE, 2)
            self.weather_grid[r][c] = rng.uniform(0.2, 1.0)

    def reset(self, seed=None):
        if seed is not None:
            self.np_rng = np.random.RandomState(seed)
        self.energy = 100.0
        self.sat_bonus = 0.0
        self.fare_earned = 0.0
        self.service_bonus = 0.0
        self._generate_hazards(seed)
        return super().reset(seed)

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        original_reward = reward
        event = info.get('event', 'move')

        if event == 'move':
            reward = -0.65
        elif event == 'wall':
            reward = -2.0
        elif event == 'pickup':
            reward = 4.0
        elif event == 'dropoff':
            pickup_row, pickup_col = LOCS[self.pass_loc]
            dest_row, dest_col = LOCS[self.destination]
            trip_distance = abs(dest_row - pickup_row) + abs(dest_col - pickup_col)
            time_cost = self.steps * 0.35
            self.fare_earned = 14.0 + trip_distance * 3.25
            reward = max(8.0, self.fare_earned - time_cost)
        elif event in ('illegal_pickup', 'illegal_dropoff'):
            reward = -15.0

        self.total_reward += reward - original_reward

        r, c = self.taxi_row, self.taxi_col
        traffic_pen = self.traffic_grid[r][c] * 1.8
        weather_pen = self.weather_grid[r][c] * 1.25
        energy_cost = 0.85 + self.traffic_grid[r][c] * 0.9 + self.weather_grid[r][c] * 0.75

        reward -= traffic_pen
        reward -= weather_pen
        self.energy = max(0.0, self.energy - energy_cost)
        self.total_reward = self.total_reward - traffic_pen - weather_pen

        if self.energy <= 0:
            reward -= 25
            self.total_reward -= 25
            done = True
            self.done = True
            info['event'] = 'energy_depleted'

        if done and self.pickup_time is not None and self.dropoff_time is not None:
            wait_penalty = self.pickup_time * 0.2
            ride_penalty = (self.dropoff_time - self.pickup_time) * 0.1
            speed_bonus = max(0.0, 8.0 - self.steps * 0.35)
            self.service_bonus = max(-6.0, 6.0 - (wait_penalty + ride_penalty))
            self.sat_bonus = speed_bonus + self.service_bonus
            reward += self.sat_bonus
            self.total_reward += self.sat_bonus

        info.update({
            'traffic_penalty': round(traffic_pen, 2),
            'weather_penalty': round(weather_pen, 2),
            'energy': round(self.energy, 2),
            'energy_cost': round(energy_cost, 2),
            'fare_earned': round(self.fare_earned, 2),
            'service_bonus': round(self.service_bonus, 2),
            'sat_bonus': round(self.sat_bonus, 2),
            'total_reward': round(self.total_reward, 2),
            'reward': round(reward, 2),
        })
        return next_state, reward, done, info

    def get_grid_state(self):
        """Return full grid state for frontend rendering."""
        cells = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cells.append({
                    'row': r,
                    'col': c,
                    'traffic': round(float(self.traffic_grid[r][c]), 2),
                    'weather': round(float(self.weather_grid[r][c]), 2),
                })

        return {
            'cells': cells,
            'taxi': {'row': self.taxi_row, 'col': self.taxi_col},
            'pass_loc': self.pass_loc,
            'pass_on_board': self.pass_on_board,
            'destination': self.destination,
            'pass_label': LOC_LABELS[self.pass_loc],
            'dest_label': LOC_LABELS[self.destination],
            'locs': [
                {'row': r, 'col': c, 'label': LOC_LABELS[i], 'color': LOC_COLORS[i]}
                for i, (r, c) in enumerate(LOCS)
            ],
            'energy': round(self.energy, 2),
            'total_reward': round(self.total_reward, 2),
            'steps': self.steps,
            'done': self.done,
        }


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.999, epsilon_min=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.trained = False
        self.episode_rewards = []
        self.eval_rewards = []
        self.eval_steps = []

    def choose_action(self, state, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        num_episodes=10000,
        max_steps=200,
        progress_callback=None,
        env_cls=TaxiEnv,
        guided_action=None,
        reward_shaper=None,
        eval_episodes=100,
    ):
        env = env_cls()
        self.episode_rewards = []
        self.eval_rewards = []
        self.eval_steps = []

        for ep in range(num_episodes):
            state = env.reset()
            total_r = 0
            for _ in range(max_steps):
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(NUM_ACTIONS)
                elif guided_action is not None:
                    action = int(guided_action(env))
                else:
                    action = int(np.argmax(self.q_table[state]))
                next_state, reward, done, _ = env.step(action)
                learning_reward = reward
                if reward_shaper is not None:
                    learning_reward = reward_shaper(env, action, reward, next_state)
                self.update(state, action, learning_reward, next_state)
                state = next_state
                total_r += reward
                if done:
                    break
            self.decay_epsilon()
            self.episode_rewards.append(total_r)

            if progress_callback and ep % 500 == 0:
                progress_callback(ep, num_episodes, total_r)

        self.trained = True
        self.epsilon = self.epsilon_min
        self.evaluate_policy(
            num_episodes=eval_episodes,
            max_steps=max_steps,
            env_cls=env_cls,
            guided_action=guided_action,
        )

    def evaluate_policy(self, num_episodes=100, max_steps=200, env_cls=TaxiEnv, guided_action=None):
        rewards = []
        steps = []
        for i in range(num_episodes):
            env = env_cls(seed=i * 137 + 42)
            state = encode_state(
                env.taxi_row,
                env.taxi_col,
                NUM_LOCATIONS if env.pass_on_board else env.pass_loc,
                env.destination,
            )
            for _ in range(max_steps):
                action = int(guided_action(env)) if guided_action is not None else self.get_best_action(state)
                state, _, done, _ = env.step(action)
                if done:
                    break
            rewards.append(float(env.total_reward))
            steps.append(int(env.steps))

        self.eval_rewards = rewards
        self.eval_steps = steps
        return rewards

    def get_best_action(self, state):
        return int(np.argmax(self.q_table[state]))

    def q_table_summary(self, n=64):
        """Return first n rows of Q-table for heatmap."""
        rows = []
        for i in range(min(n, NUM_STATES)):
            row = self.q_table[i].tolist()
            rows.append({
                'state': i,
                'q_values': [round(v, 3) for v in row],
                'best_action': int(np.argmax(row)),
                'max_q': round(float(np.max(row)), 3),
            })
        return rows

    def training_stats(self):
        rewards = self.episode_rewards
        if not rewards:
            return {}
        policy_rewards = self.eval_rewards or []
        return {
            'total_episodes': len(rewards),
            'final_epsilon': round(self.epsilon, 4),
            'avg_reward_last_100': round(float(np.mean(rewards[-100:])), 2),
            'avg_reward_overall': round(float(np.mean(rewards)), 2),
            'policy_eval_episodes': len(self.eval_rewards),
            'policy_avg_steps': round(float(np.mean(self.eval_steps)), 1) if self.eval_steps else None,
            'policy_avg_reward': round(float(np.mean(policy_rewards)), 2) if policy_rewards else None,
            'policy_max_reward': round(float(np.max(policy_rewards)), 2) if policy_rewards else None,
            'policy_min_reward': round(float(np.min(policy_rewards)), 2) if policy_rewards else None,
            'max_reward': round(float(np.max(rewards)), 2),
            'min_reward': round(float(np.min(rewards)), 2),
            'reward_history_sampled': [
                round(float(v), 2) for v in rewards[::max(1, len(rewards) // 200)]
            ],
        }
