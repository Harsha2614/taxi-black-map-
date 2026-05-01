"""
Small NumPy graph convolution model for the taxi grid.

The model learns a per-cell traversal cost from graph-structured episode
experience. It is intentionally dependency-light so the demo can run without
PyTorch.
"""
import numpy as np

from taxi_env import GRID_SIZE, LOCS, MOVES


NUM_NODES = GRID_SIZE * GRID_SIZE


def _node_index(row, col):
    return row * GRID_SIZE + col


def _build_normalized_adjacency():
    adj = np.eye(NUM_NODES, dtype=float)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            src = _node_index(row, col)
            for dr, dc in MOVES:
                nr, nc = row + dr, col + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    adj[src, _node_index(nr, nc)] = 1.0

    degree = np.sum(adj, axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1.0)))
    return inv_sqrt @ adj @ inv_sqrt


class TaxiGCNCostModel:
    """Two-layer GCN that predicts graph-aware movement cost per grid cell."""

    def __init__(self, hidden_dim=16, learning_rate=0.03, seed=7):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed)
        self.adj = _build_normalized_adjacency()
        self.input_dim = 9
        self.w0 = self.rng.normal(0.0, 0.18, (self.input_dim, hidden_dim))
        self.b0 = np.zeros(hidden_dim)
        self.w1 = self.rng.normal(0.0, 0.18, (hidden_dim, 1))
        self.b1 = np.zeros(1)
        self.trained = False
        self.loss_history = []
        self.target_source = 'untrained'
        self.transition_samples = 0

    def features_for(self, env, target_row, target_col):
        features = []
        pickup_row, pickup_col = LOCS[env.pass_loc]
        dest_row, dest_col = LOCS[env.destination]
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                features.append([
                    float(env.traffic_grid[row][col]),
                    float(env.weather_grid[row][col]),
                    1.0 if (row, col) == (env.taxi_row, env.taxi_col) else 0.0,
                    1.0 if (row, col) == (target_row, target_col) else 0.0,
                    1.0 if (row, col) == (pickup_row, pickup_col) else 0.0,
                    1.0 if (row, col) == (dest_row, dest_col) else 0.0,
                    row / (GRID_SIZE - 1),
                    col / (GRID_SIZE - 1),
                    1.0 if env.pass_on_board else 0.0,
                ])
        return np.asarray(features, dtype=float)

    def target_costs_for(self, env, target_row, target_col):
        """Formula baseline kept for comparison; training now uses rewards."""
        costs = []
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                traffic = float(env.traffic_grid[row][col])
                weather = float(env.weather_grid[row][col])
                dist = abs(target_row - row) + abs(target_col - col)
                hazard = traffic * 2.0 + weather * 1.5
                risk = (traffic ** 2) * 5.0 + (weather ** 2) * 4.0
                severe = 0.0
                if traffic >= 0.55:
                    severe += 4.0 + traffic * 5.0
                if weather >= 0.50:
                    severe += 3.0 + weather * 4.0
                costs.append(hazard + risk + severe + dist * 0.35)

        arr = np.asarray(costs, dtype=float)
        arr[_node_index(target_row, target_col)] *= 0.25
        return arr

    def forward(self, x):
        ax = self.adj @ x
        z0 = ax @ self.w0 + self.b0
        h = np.maximum(z0, 0.0)
        ah = self.adj @ h
        y = (ah @ self.w1 + self.b1).reshape(-1)
        return y, (x, ax, z0, h, ah)

    def transition_cost_target(self, env, target_row, target_col, row, col, reward):
        """Build a masked target from one experienced transition reward."""
        features = self.features_for(env, target_row, target_col)
        return self.transition_cost_sample(features, row, col, reward)

    def transition_cost_sample(self, features, row, col, reward):
        """Build a masked target for an already-captured feature tensor."""
        target = np.zeros(NUM_NODES, dtype=float)
        mask = np.zeros(NUM_NODES, dtype=float)
        idx = _node_index(row, col)
        target[idx] = np.clip(-float(reward), 0.01, 40.0)
        mask[idx] = 1.0
        return features, target, mask

    def fit(self, samples, epochs=120, target_source='supervised'):
        self.loss_history = []
        for _ in range(epochs):
            total_loss = 0.0
            for sample in samples:
                if len(sample) == 3:
                    x, target, mask = sample
                    mask = mask.reshape(NUM_NODES)
                    denom = max(float(np.sum(mask)), 1.0)
                else:
                    x, target = sample
                    mask = np.ones(NUM_NODES, dtype=float)
                    denom = float(NUM_NODES)

                pred, cache = self.forward(x)
                err = (pred - target) * mask
                total_loss += float(np.sum(err ** 2) / denom)

                grad_y = (2.0 / denom) * err.reshape(NUM_NODES, 1)
                x0, ax, z0, h, ah = cache
                grad_w1 = ah.T @ grad_y
                grad_b1 = np.sum(grad_y, axis=0)
                grad_ah = grad_y @ self.w1.T
                grad_h = self.adj.T @ grad_ah
                grad_z0 = grad_h * (z0 > 0)
                grad_w0 = ax.T @ grad_z0
                grad_b0 = np.sum(grad_z0, axis=0)

                self.w1 -= self.learning_rate * grad_w1
                self.b1 -= self.learning_rate * grad_b1
                self.w0 -= self.learning_rate * grad_w0
                self.b0 -= self.learning_rate * grad_b0

            self.loss_history.append(total_loss / max(len(samples), 1))

        self.trained = True
        self.target_source = target_source
        self.transition_samples = len(samples)
        return self.loss_history

    def predict_costs(self, env, target_row, target_col):
        x = self.features_for(env, target_row, target_col)
        pred, _ = self.forward(x)
        pred = np.maximum(pred, 0.01)
        return pred.reshape(GRID_SIZE, GRID_SIZE)

    def summary(self):
        if not self.loss_history:
            return {'trained': self.trained}
        return {
            'trained': self.trained,
            'epochs': len(self.loss_history),
            'initial_loss': round(float(self.loss_history[0]), 4),
            'final_loss': round(float(self.loss_history[-1]), 4),
            'target_source': self.target_source,
            'transition_samples': self.transition_samples,
        }

    def save(self, path):
        np.savez(
            path,
            w0=self.w0,
            b0=self.b0,
            w1=self.w1,
            b1=self.b1,
            loss_history=np.asarray(self.loss_history, dtype=float),
            trained=np.asarray([self.trained], dtype=bool),
            target_source=np.asarray([self.target_source]),
            transition_samples=np.asarray([self.transition_samples], dtype=int),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        model = cls(hidden_dim=int(data['w0'].shape[1]))
        model.w0 = data['w0']
        model.b0 = data['b0']
        model.w1 = data['w1']
        model.b1 = data['b1']
        model.loss_history = data['loss_history'].tolist()
        model.trained = bool(data['trained'][0])
        if 'target_source' in data:
            model.target_source = str(data['target_source'][0])
        if 'transition_samples' in data:
            model.transition_samples = int(data['transition_samples'][0])
        return model
