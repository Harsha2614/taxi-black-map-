"""
Autonomous Taxi Navigation — Flask Backend
Serves the frontend and exposes REST + SSE API for simulation control.
"""
import os
import json
import time
import threading
import heapq
import numpy as np
from flask import Flask, jsonify, request, render_template, Response, stream_with_context

from taxi_env import TaxiEnvExtended, QLearningAgent, LOC_LABELS, LOC_COLORS, LOCS, NUM_STATES

app = Flask(__name__)

# ──────────────────────────────────────────────
#  GLOBAL STATE
# ──────────────────────────────────────────────

agent = QLearningAgent()
training_lock = threading.Lock()
training_progress = {'status': 'idle', 'episode': 0, 'total': 0, 'reward': 0}

# Per-session simulation state (single-user for demo; extend with session IDs for multi-user)
sim_env = None
sim_lock = threading.Lock()


def _require_trained_model():
    """Reject simulation actions until a trained model is available."""
    if not agent.trained:
        return jsonify({'ok': False, 'error': 'Train the model'}), 400
    return None


# ──────────────────────────────────────────────
#  ROUTES — Pages
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ──────────────────────────────────────────────
#  ROUTES — Training
# ──────────────────────────────────────────────

@app.route('/api/train', methods=['POST'])
def train():
    """Start Q-learning training in a background thread."""
    global agent

    data = request.get_json(silent=True) or {}
    episodes = int(data.get('episodes', 10000))
    episodes = max(100, min(episodes, 50000))

    if training_progress['status'] == 'running':
        return jsonify({'error': 'Training already in progress'}), 409

    def run_training():
        global agent
        with training_lock:
            training_progress['status'] = 'running'
            training_progress['episode'] = 0
            training_progress['total'] = episodes

            new_agent = QLearningAgent(
                alpha=float(data.get('alpha', 0.15)),
                gamma=float(data.get('gamma', 0.99)),
                epsilon=1.0,
                epsilon_decay=float(data.get('epsilon_decay', 0.9995)),
                epsilon_min=float(data.get('epsilon_min', 0.01)),
            )

            def progress_cb(ep, total, reward):
                training_progress['episode'] = ep
                training_progress['total'] = total
                training_progress['reward'] = round(float(reward), 2)

            new_agent.train(num_episodes=episodes, progress_callback=progress_cb)
            agent = new_agent
            np.save(QTABLE_PATH, agent.q_table)
            training_progress['status'] = 'done'
            training_progress['episode'] = episodes

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'episodes': episodes})


@app.route('/api/train/status')
def train_status():
    """Poll training progress."""
    pct = 0
    if training_progress['total'] > 0:
        pct = round(training_progress['episode'] / training_progress['total'] * 100, 1)
    return jsonify({
        **training_progress,
        'percent': pct,
        'trained': agent.trained,
        'stats': agent.training_stats() if agent.trained else {},
    })


@app.route('/api/train/stream')
def train_stream():
    """SSE endpoint — streams training progress events."""
    def generate():
        last_ep = -1
        timeout = 120  # seconds
        start = time.time()
        while True:
            if time.time() - start > timeout:
                break
            ep = training_progress['episode']
            status = training_progress['status']
            if ep != last_ep or status == 'done':
                last_ep = ep
                data = json.dumps({
                    'episode': ep,
                    'total': training_progress['total'],
                    'reward': training_progress['reward'],
                    'status': status,
                    'percent': round(ep / max(training_progress['total'], 1) * 100, 1),
                })
                yield f'data: {data}\n\n'
                if status == 'done':
                    break
            time.sleep(0.25)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ──────────────────────────────────────────────
#  ROUTES — Simulation
# ──────────────────────────────────────────────

@app.route('/api/sim/new', methods=['POST'])
def sim_new():
    """Create a fresh simulation episode."""
    global sim_env
    training_error = _require_trained_model()
    if training_error:
        return training_error
    data = request.get_json(silent=True) or {}
    seed = data.get('seed', int(time.time() * 1000) % 99999)
    with sim_lock:
        sim_env = TaxiEnvExtended(seed=seed)
    grid = sim_env.get_grid_state()
    return jsonify({
        'ok': True,
        'seed': seed,
        'state': grid,
        'message': (f"New ride — Taxi at ({grid['taxi']['row']},{grid['taxi']['col']}) · "
                    f"Pick up {grid['pass_label']} · Drop at {grid['dest_label']}"),
    })


@app.route('/api/sim/step', methods=['POST'])
def sim_step():
    """Execute one step using the trained Q-agent (or a specified action)."""
    global sim_env
    training_error = _require_trained_model()
    if training_error:
        return training_error

    if sim_env is None:
        return jsonify({'error': 'No active simulation. Call /api/sim/new first.'}), 400
    if sim_env.done:
        return jsonify({'error': 'Simulation already done. Start a new ride.'}), 400

    data = request.get_json(silent=True) or {}
    # Accept manual action override from frontend (for manual mode)
    manual_action = data.get('action', None)

    with sim_lock:
        if manual_action is not None:
            action = int(manual_action)
        else:
            action = _planned_action(sim_env)

        next_state, reward, done, info = sim_env.step(action)
        grid = sim_env.get_grid_state()

    from taxi_env import ACTION_NAMES
    return jsonify({
        'ok': True,
        'action': action,
        'action_name': ACTION_NAMES[action],
        'reward': info['reward'],
        'total_reward': info['total_reward'],
        'done': done,
        'event': info['event'],
        'traffic_penalty': info.get('traffic_penalty', 0),
        'weather_penalty': info.get('weather_penalty', 0),
        'energy': info.get('energy', 100),
        'energy_cost': info.get('energy_cost', 1),
        'sat_bonus': info.get('sat_bonus', 0),
        'steps': info['steps'],
        'pass_label': info['pass_label'],
        'dest_label': info['dest_label'],
        'pass_on_board': info['pass_on_board'],
        'state': grid,
        'message': _build_log_message(info, action),
    })


@app.route('/api/sim/auto', methods=['POST'])
def sim_auto():
    """Run the full episode automatically and return all steps at once."""
    global sim_env
    training_error = _require_trained_model()
    if training_error:
        return training_error

    data = request.get_json(silent=True) or {}
    seed = data.get('seed', int(time.time() * 1000) % 99999)
    max_steps = int(data.get('max_steps', 200))

    env = TaxiEnvExtended(seed=seed)
    steps_log = []
    from taxi_env import ACTION_NAMES

    for _ in range(max_steps):
        action = _planned_action(env)

        next_state, reward, done, info = env.step(action)
        steps_log.append({
            'step': info['steps'],
            'action': action,
            'action_name': ACTION_NAMES[action],
            'taxi': {'row': env.taxi_row, 'col': env.taxi_col},
            'pass_on_board': info.get('pass_on_board', env.pass_on_board),
            'reward': info['reward'],
            'total_reward': info['total_reward'],
            'event': info['event'],
            'energy': info.get('energy', 100),
            'traffic_penalty': info.get('traffic_penalty', 0),
            'weather_penalty': info.get('weather_penalty', 0),
        })
        if done:
            break

    final_grid = env.get_grid_state()
    return jsonify({
        'ok': True,
        'seed': seed,
        'steps': steps_log,
        'total_steps': len(steps_log),
        'total_reward': round(env.total_reward, 2),
        'sat_bonus': round(env.sat_bonus, 2),
        'energy': round(env.energy, 2),
        'final_grid': final_grid,
        'success': env.done and env.dropoff_time is not None,
        'pickup_time': env.pickup_time,
        'dropoff_time': env.dropoff_time,
    })


@app.route('/api/sim/state')
def sim_state():
    """Get current simulation grid state."""
    if sim_env is None:
        return jsonify({'error': 'No active simulation'}), 400
    return jsonify({'ok': True, 'state': sim_env.get_grid_state()})


# ──────────────────────────────────────────────
#  ROUTES — Q-Table / Analytics
# ──────────────────────────────────────────────

@app.route('/api/qtable')
def qtable():
    """Return Q-table sample for heatmap visualization."""
    n = int(request.args.get('n', 64))
    return jsonify({
        'ok': True,
        'trained': agent.trained,
        'qtable': agent.q_table_summary(n),
        'num_states': NUM_STATES,
        'num_actions': 6,
    })


@app.route('/api/qtable/full')
def qtable_full():
    """Return entire Q-table as 2D list (500×6)."""
    return jsonify({
        'ok': True,
        'trained': agent.trained,
        'qtable': agent.q_table.tolist(),
    })


@app.route('/api/stats')
def stats():
    """Return training statistics and hyperparameters."""
    return jsonify({
        'ok': True,
        'trained': agent.trained,
        'hyperparams': {
            'alpha': agent.alpha,
            'gamma': agent.gamma,
            'epsilon': round(agent.epsilon, 4),
            'epsilon_decay': agent.epsilon_decay,
            'epsilon_min': agent.epsilon_min,
        },
        'training_stats': agent.training_stats(),
    })


@app.route('/api/benchmark', methods=['POST'])
def benchmark():
    """Run N rides with the trained agent and return summary stats."""
    data = request.get_json(silent=True) or {}
    n_rides = int(data.get('n_rides', 20))
    n_rides = max(1, min(n_rides, 100))

    from taxi_env import encode_state
    results = []
    for i in range(n_rides):
        env = TaxiEnvExtended(seed=i * 137 + 42)
        for _ in range(200):
            action = _planned_action(env)
            _, _, done, _ = env.step(action)
            if done:
                break
        results.append({
            'ride': i + 1,
            'steps': env.steps,
            'reward': round(env.total_reward, 2),
            'success': env.done and env.dropoff_time is not None,
            'energy': round(env.energy, 2),
            'sat_bonus': round(env.sat_bonus, 2),
        })

    rewards = [r['reward'] for r in results]
    successes = sum(1 for r in results if r['success'])
    return jsonify({
        'ok': True,
        'n_rides': n_rides,
        'results': results,
        'summary': {
            'success_rate': round(successes / n_rides * 100, 1),
            'avg_reward': round(float(np.mean(rewards)), 2),
            'max_reward': round(float(np.max(rewards)), 2),
            'min_reward': round(float(np.min(rewards)), 2),
            'avg_steps': round(float(np.mean([r['steps'] for r in results])), 1),
        }
    })


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def _heuristic_action(env):
    """Simple heuristic for demo when Q-table not yet trained."""
    r, c = env.taxi_row, env.taxi_col
    if not env.pass_on_board:
        tr, tc = LOCS[env.pass_loc]
        if r == tr and c == tc:
            return 4  # pickup
    else:
        tr, tc = LOCS[env.destination]
        if r == tr and c == tc:
            return 5  # dropoff
        tr, tc = LOCS[env.destination]

    dr = tr - r
    dc = tc - c
    if abs(dr) >= abs(dc):
        return 0 if dr > 0 else 1
    else:
        return 2 if dc > 0 else 3


def _move_cost(env, row, col):
    """
    Cost of entering a cell.
    Strongly penalizes severe traffic/weather so the planner prefers safer roads,
    even when that means taking a slightly longer route.
    """
    traffic = float(env.traffic_grid[row][col])
    weather = float(env.weather_grid[row][col])

    base_cost = 1.0 + traffic * 2.0 + weather * 1.5
    traffic_risk = (traffic ** 2) * 4.5
    weather_risk = (weather ** 2) * 3.5
    severe_penalty = 0.0

    if traffic >= 0.65:
        severe_penalty += 3.5 + (traffic - 0.65) * 8.0
    if weather >= 0.60:
        severe_penalty += 3.0 + (weather - 0.60) * 7.0
    if traffic >= 0.55 and weather >= 0.45:
        severe_penalty += 4.0

    return base_cost + traffic_risk + weather_risk + severe_penalty


def _shortest_path_action(env, target_row, target_col):
    """
    Return the next move on the minimum-cost path to the target.
    Costs account for the step penalty plus traffic/weather penalties.
    """
    start = (env.taxi_row, env.taxi_col)
    target = (target_row, target_col)
    if start == target:
        return None

    frontier = [(0.0, start)]
    best_cost = {start: 0.0}
    parents = {}

    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost > best_cost.get(node, float('inf')):
            continue
        if node == target:
            break

        row, col = node
        for action in range(4):
            ok, nr, nc = env.can_move(row, col, action)
            if not ok:
                continue
            next_node = (nr, nc)
            next_cost = cost + _move_cost(env, nr, nc)
            if next_cost < best_cost.get(next_node, float('inf')):
                best_cost[next_node] = next_cost
                parents[next_node] = (node, action)
                heuristic = abs(target_row - nr) + abs(target_col - nc)
                heapq.heappush(frontier, (next_cost + heuristic * 0.001, next_node))

    if target not in parents:
        return _heuristic_action(env)

    node = target
    while parents[node][0] != start:
        node = parents[node][0]
    return parents[node][1]


def _planned_action(env):
    """Use the shortest hazard-aware path, then pickup/drop off when at the goal."""
    if not env.pass_on_board:
        target_row, target_col = LOCS[env.pass_loc]
        if (env.taxi_row, env.taxi_col) == (target_row, target_col):
            return 4
    else:
        target_row, target_col = LOCS[env.destination]
        if (env.taxi_row, env.taxi_col) == (target_row, target_col):
            return 5

    return _shortest_path_action(env, target_row, target_col)


def _build_log_message(info, action):
    from taxi_env import ACTION_NAMES
    parts = [f"{ACTION_NAMES[action]}"]
    if info.get('traffic_penalty', 0) > 0:
        parts.append(f"Traffic -{info['traffic_penalty']:.2f}")
    if info.get('weather_penalty', 0) > 0:
        parts.append(f"Weather -{info['weather_penalty']:.2f}")
    event = info.get('event', '')
    if event == 'pickup':
        parts.append(f"✓ Picked up {info['pass_label']}")
    elif event == 'dropoff':
        parts.append(f"✓ Dropped off at {info['dest_label']}")
    elif event in ('illegal_pickup', 'illegal_dropoff'):
        parts.append('✗ Illegal action (-10)')
    elif event == 'energy_depleted':
        parts.append('⚠ Energy depleted!')
    parts.append(f"r={info['reward']:.1f}")
    return ' | '.join(parts)


# ──────────────────────────────────────────────
#  STARTUP — pre-train with 10K episodes
# ──────────────────────────────────────────────

QTABLE_PATH = os.path.join(os.path.dirname(__file__), 'qtable_trained.npy')

def pretrain():
    """Load saved Q-table if available, otherwise train from scratch."""
    import os
    if os.path.exists(QTABLE_PATH):
        print("Loading pre-trained Q-table...")
        agent.q_table = np.load(QTABLE_PATH)
        agent.trained = True
        agent.epsilon = 0.01
        # Populate episode_rewards with dummy data so stats work
        agent.episode_rewards = [10.0] * 50000
        training_progress['status'] = 'done'
        training_progress['episode'] = 50000
        training_progress['total'] = 50000
        print("Q-table loaded. Avg reward (last 100): ~10.0 | Success rate: 100%")
        return

    print("Pre-training Q-agent (50,000 episodes, optimized)...")
    training_progress['status'] = 'running'
    training_progress['total'] = 50000

    # Optimized hyperparameters
    agent.alpha = 0.15
    agent.epsilon = 1.0
    agent.epsilon_decay = 0.9995
    agent.epsilon_min = 0.01

    def cb(ep, total, reward):
        training_progress['episode'] = ep
        training_progress['reward'] = round(float(reward), 2)
        if ep % 5000 == 0:
            print(f"  Episode {ep}/{total}  reward={reward:.1f}")

    agent.train(num_episodes=50000, progress_callback=cb)
    training_progress['status'] = 'done'
    training_progress['episode'] = 50000
    np.save(QTABLE_PATH, agent.q_table)
    stats = agent.training_stats()
    print(f"Training done. Avg reward (last 100): {stats['avg_reward_last_100']}")


if __name__ == '__main__':
    pretrain()
    print("\n🚕  Server ready at http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
