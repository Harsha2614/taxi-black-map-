"""
Autonomous Taxi Navigation — Flask Backend
Serves the frontend and exposes REST + SSE API for simulation control.
"""
import os
import json
import time
import threading
import heapq
import urllib.parse
import urllib.request
import urllib.error
import numpy as np
from flask import Flask, jsonify, request, render_template, Response, stream_with_context

from taxi_env import TaxiEnvExtended, QLearningAgent, LOC_LABELS, LOC_COLORS, LOCS, NUM_STATES, ACTION_NAMES, GRID_SIZE
from gcn_model import TaxiGCNCostModel

try:
    import mysql.connector
except ImportError:
    mysql = None
else:
    mysql = mysql.connector

app = Flask(__name__)


def _load_local_env(path='.env'):
    """Load simple KEY=value pairs for local Flask runs."""
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()

# ──────────────────────────────────────────────
#  GLOBAL STATE
# ──────────────────────────────────────────────

agent = QLearningAgent()
gcn_model = None
training_lock = threading.Lock()
training_progress = {'status': 'idle', 'episode': 0, 'total': 0, 'reward': 0}

# Per-session simulation state (single-user for demo; extend with session IDs for multi-user)
sim_env = None
sim_lock = threading.Lock()
route_geometry_cache = {}
db_lock = threading.Lock()
db_initialized = False
db_error = None

# Hyderabad sample points matching the frontend's map grid.
MAP_LAT_MIN = 17.330
MAP_LAT_MAX = 17.470
MAP_LNG_MIN = 78.420
MAP_LNG_MAX = 78.520
ROAD_NODE_OFFSETS = [
    [(0.0020, -0.0010), (0.0040, 0.0040), (0.0025, 0.0080), (0.0010, 0.0120), (0.0000, 0.0000)],
    [(0.0040, -0.0030), (0.0020, 0.0010), (0.0010, 0.0060), (0.0020, 0.0100), (0.0030, 0.0130)],
    [(0.0030, -0.0050), (0.0010, -0.0010), (0.0000, 0.0030), (-0.0010, 0.0070), (0.0010, 0.0100)],
    [(0.0010, -0.0060), (-0.0010, -0.0030), (-0.0020, 0.0020), (-0.0010, 0.0050), (0.0000, 0.0080)],
    [(0.0000, 0.0000), (-0.0030, -0.0040), (-0.0020, 0.0000), (0.0000, 0.0000), (0.0010, 0.0050)],
]
MAP_LOCS_ADJUSTED = [
    {'lat': 17.361, 'lng': 78.474},
    {'lat': 17.450, 'lng': 78.381},
    {'lat': 17.440, 'lng': 78.500},
    {'lat': 17.416, 'lng': 78.448},
    {'lat': 17.400, 'lng': 78.473},
    {'lat': 17.441, 'lng': 78.446},
    {'lat': 17.432, 'lng': 78.407},
    {'lat': 17.444, 'lng': 78.487},
    {'lat': 17.395, 'lng': 78.430},
    {'lat': 17.337, 'lng': 78.520},
    {'lat': 17.389, 'lng': 78.486},
    {'lat': 17.422, 'lng': 78.475},
    {'lat': 17.448, 'lng': 78.391},
    {'lat': 17.437, 'lng': 78.457},
    {'lat': 17.391, 'lng': 78.490},
    {'lat': 17.470, 'lng': 78.449},
    {'lat': 17.439, 'lng': 78.417},
    {'lat': 17.436, 'lng': 78.458},
    {'lat': 17.400, 'lng': 78.418},
    {'lat': 17.403, 'lng': 78.530},
    {'lat': 17.367, 'lng': 78.414},
    {'lat': 17.363, 'lng': 78.472},
    {'lat': 17.366, 'lng': 78.492},
    {'lat': 17.329, 'lng': 78.441},
    {'lat': 17.328, 'lng': 78.470},
]


def _require_trained_model():
    """Reject simulation actions until a trained model is available."""
    if not agent.trained:
        return jsonify({'ok': False, 'error': 'Train the model'}), 400
    return None


def _mysql_config(include_database=True):
    database = os.getenv('MYSQL_DATABASE')
    user = os.getenv('MYSQL_USER')
    if not database or not user:
        return None
    config = {
        'host': os.getenv('MYSQL_HOST', '127.0.0.1'),
        'port': int(os.getenv('MYSQL_PORT', '3306')),
        'user': user,
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'connection_timeout': 2,
    }
    if include_database:
        config['database'] = database
    return config


def _mysql_connection(include_database=True):
    if mysql is None:
        raise RuntimeError('mysql-connector-python is not installed')
    config = _mysql_config(include_database=include_database)
    if not config:
        raise RuntimeError('Missing MySQL env vars: MYSQL_USER and MYSQL_DATABASE')
    return mysql.connect(**config)


def _ensure_ride_history_table():
    """Create the MySQL database/table used for persisted ride history."""
    global db_initialized, db_error
    if db_initialized:
        return True
    with db_lock:
        if db_initialized:
            return True
        try:
            database = os.getenv('MYSQL_DATABASE')
            if not database:
                raise RuntimeError('Missing MYSQL_DATABASE')
            root_conn = _mysql_connection(include_database=False)
            root_cursor = root_conn.cursor()
            root_cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            root_conn.commit()
            root_cursor.close()
            root_conn.close()

            conn = _mysql_connection(include_database=True)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ride_history (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    ride_id VARCHAR(40) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mode VARCHAR(24) NOT NULL,
                    pickup_index INT NOT NULL,
                    pickup_label VARCHAR(8) NOT NULL,
                    dropoff_index INT NOT NULL,
                    dropoff_label VARCHAR(8) NOT NULL,
                    hazard_source VARCHAR(80),
                    success BOOLEAN NOT NULL,
                    total_reward DOUBLE NOT NULL,
                    total_steps INT NOT NULL,
                    energy_remaining DOUBLE,
                    satisfaction_bonus DOUBLE,
                    pickup_time INT,
                    dropoff_time INT,
                    route_json LONGTEXT,
                    steps_json LONGTEXT,
                    live_warnings_json LONGTEXT,
                    live_samples_json LONGTEXT,
                    map_details_json LONGTEXT,
                    INDEX idx_created_at (created_at),
                    INDEX idx_success (success),
                    INDEX idx_mode (mode)
                )
            """)
            _ensure_ride_history_column(cursor, database, 'live_samples_json', 'LONGTEXT')
            _ensure_ride_history_column(cursor, database, 'map_details_json', 'LONGTEXT')
            conn.commit()
            cursor.close()
            conn.close()
            db_initialized = True
            db_error = None
            return True
        except Exception as exc:
            db_error = str(exc)
            print(f"MySQL ride history unavailable: {db_error}", flush=True)
            return False


def _ensure_ride_history_column(cursor, database, column_name, column_type):
    """Add a ride_history column when upgrading an existing local database."""
    cursor.execute("""
        SELECT COUNT(*)
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'ride_history' AND COLUMN_NAME = %s
    """, (database, column_name))
    exists = cursor.fetchone()[0] > 0
    if not exists:
        cursor.execute(f"ALTER TABLE ride_history ADD COLUMN {column_name} {column_type}")


def _map_details_for_history(env, route):
    """Build a compact JSON payload with map-specific ride details."""
    cells = []
    for cell in env.get_grid_state().get('cells', []):
        lat, lng = _grid_lat_lng(cell['row'], cell['col'])
        cells.append({
            'row': cell['row'],
            'col': cell['col'],
            'lat': round(lat, 6),
            'lng': round(lng, 6),
            'traffic': cell.get('traffic', 0),
            'weather': cell.get('weather', 0),
        })

    pickup_map = MAP_LOCS_ADJUSTED[env.pass_loc] if env.pass_loc < len(MAP_LOCS_ADJUSTED) else {}
    dropoff_map = MAP_LOCS_ADJUSTED[env.destination] if env.destination < len(MAP_LOCS_ADJUSTED) else {}
    return {
        'grid_size': GRID_SIZE,
        'pickup': {
            'index': env.pass_loc,
            'label': LOC_LABELS[env.pass_loc],
            'grid': {'row': LOCS[env.pass_loc][0], 'col': LOCS[env.pass_loc][1]},
            'map': pickup_map,
        },
        'dropoff': {
            'index': env.destination,
            'label': LOC_LABELS[env.destination],
            'grid': {'row': LOCS[env.destination][0], 'col': LOCS[env.destination][1]},
            'map': dropoff_map,
        },
        'final_taxi': {'row': env.taxi_row, 'col': env.taxi_col},
        'route': route,
        'cells': cells,
        'hazard_source': getattr(env, 'hazard_source', 'generated'),
        'hazards_fetched_at': getattr(env, 'hazards_fetched_at', None),
    }


def _record_ride_history(env, mode, hazard_source=None, steps_log=None):
    """Persist a completed ride to MySQL when configured."""
    if getattr(env, 'ride_recorded', False):
        return False
    if not _ensure_ride_history_table():
        return False
    route = _planned_route(env)
    live_warnings = getattr(env, 'live_warnings', [])
    live_samples = getattr(env, 'live_samples', [])
    if steps_log is None:
        steps_log = getattr(env, 'ride_steps', [])
    map_details = _map_details_for_history(env, route)
    payload = {
        'ride_id': f"ride-{int(time.time() * 1000)}",
        'mode': mode,
        'pickup_index': int(env.pass_loc),
        'pickup_label': LOC_LABELS[env.pass_loc],
        'dropoff_index': int(env.destination),
        'dropoff_label': LOC_LABELS[env.destination],
        'hazard_source': hazard_source or getattr(env, 'hazard_source', 'generated'),
        'success': bool(env.done and env.dropoff_time is not None),
        'total_reward': float(round(env.total_reward, 2)),
        'total_steps': int(env.steps),
        'energy_remaining': float(round(getattr(env, 'energy', 0.0), 2)),
        'satisfaction_bonus': float(round(getattr(env, 'sat_bonus', 0.0), 2)),
        'pickup_time': env.pickup_time,
        'dropoff_time': env.dropoff_time,
        'route_json': json.dumps(route),
        'steps_json': json.dumps(steps_log or []),
        'live_warnings_json': json.dumps(live_warnings),
        'live_samples_json': json.dumps(live_samples),
        'map_details_json': json.dumps(map_details),
    }
    try:
        conn = _mysql_connection(include_database=True)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ride_history (
                ride_id, mode, pickup_index, pickup_label, dropoff_index, dropoff_label,
                hazard_source, success, total_reward, total_steps, energy_remaining,
                satisfaction_bonus, pickup_time, dropoff_time, route_json, steps_json,
                live_warnings_json, live_samples_json, map_details_json
            )
            VALUES (
                %(ride_id)s, %(mode)s, %(pickup_index)s, %(pickup_label)s,
                %(dropoff_index)s, %(dropoff_label)s, %(hazard_source)s, %(success)s,
                %(total_reward)s, %(total_steps)s, %(energy_remaining)s,
                %(satisfaction_bonus)s, %(pickup_time)s, %(dropoff_time)s,
                %(route_json)s, %(steps_json)s, %(live_warnings_json)s,
                %(live_samples_json)s, %(map_details_json)s
            )
        """, payload)
        conn.commit()
        cursor.close()
        conn.close()
        env.ride_recorded = True
        return True
    except Exception as exc:
        print(f"MySQL ride history insert failed: {exc}", flush=True)
        return False


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
    episodes = max(100, min(episodes, 100000))

    if training_progress['status'] == 'running':
        return jsonify({'error': 'Training already in progress'}), 409

    def run_training():
        global agent, gcn_model
        with training_lock:
            training_progress['status'] = 'running'
            training_progress['episode'] = 0
            training_progress['total'] = episodes

            new_gcn = _train_gcn_model(
                samples=int(data.get('gcn_samples', 180)),
                epochs=int(data.get('gcn_epochs', 140)),
                seed=data.get('seed'),
            )
            gcn_model = new_gcn
            new_gcn.save(GCN_PATH)

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

            new_agent.train(
                num_episodes=episodes,
                progress_callback=progress_cb,
                env_cls=TaxiEnvExtended,
                guided_action=_planned_action,
                reward_shaper=_gcn_reward_shaper(new_gcn),
                eval_episodes=100,
            )
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
        'gcn': gcn_model.summary() if gcn_model else {'trained': False},
        'num_states': NUM_STATES,
        'num_actions': 6,
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
    live = bool(data.get('live', False))
    with sim_lock:
        sim_env = TaxiEnvExtended(seed=seed)
        try:
            _apply_ride_selection(sim_env, data)
            if live:
                data = {**data, **_fetch_live_hazard_payload()}
            hazard_source = _apply_external_hazards(sim_env, data)
            sim_env.ride_mode = 'live' if live else 'manual'
            sim_env.ride_recorded = False
            sim_env.ride_steps = []
        except ValueError as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 502
    grid = _serialize_state(sim_env)
    return jsonify({
        'ok': True,
        'seed': seed,
        'hazard_source': hazard_source,
        'live_warnings': getattr(sim_env, 'live_warnings', []),
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
            decision_reason = "manual override selected this action"
        else:
            action = _planned_action(sim_env)
            decision_reason = _decision_reason(sim_env, action)

        next_state, reward, done, info = sim_env.step(action)
        sim_env.ride_steps.append({
            'step': info['steps'],
            'action': action,
            'action_name': ACTION_NAMES[action],
            'taxi': {'row': sim_env.taxi_row, 'col': sim_env.taxi_col},
            'pass_on_board': info.get('pass_on_board', sim_env.pass_on_board),
            'reward': info['reward'],
            'total_reward': info['total_reward'],
            'event': info['event'],
            'energy': info.get('energy', 100),
            'traffic_penalty': info.get('traffic_penalty', 0),
            'weather_penalty': info.get('weather_penalty', 0),
            'decision_reason': decision_reason,
        })
        if done:
            _record_ride_history(
                sim_env,
                getattr(sim_env, 'ride_mode', 'manual'),
                getattr(sim_env, 'hazard_source', 'generated'),
            )
        grid = _serialize_state(sim_env)

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
        'decision_reason': decision_reason,
        'state': grid,
        'message': _build_log_message(info, action, decision_reason),
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
    live = bool(data.get('live', False))

    env = TaxiEnvExtended(seed=seed)
    try:
        _apply_ride_selection(env, data)
        if live:
            data = {**data, **_fetch_live_hazard_payload()}
        hazard_source = _apply_external_hazards(env, data)
        env.ride_mode = 'live_auto' if live else 'auto'
        env.ride_recorded = False
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 502
    steps_log = []
    from taxi_env import ACTION_NAMES

    for _ in range(max_steps):
        action = _planned_action(env)
        decision_reason = _decision_reason(env, action)

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
            'decision_reason': decision_reason,
            'message': _build_log_message(info, action, decision_reason),
            'planned_route': _planned_route(env),
        })
        if done:
            break

    if env.done:
        _record_ride_history(env, getattr(env, 'ride_mode', 'auto'), hazard_source, steps_log)

    return jsonify({
        'ok': True,
        'seed': seed,
        'hazard_source': hazard_source,
        'steps': steps_log,
        'total_steps': len(steps_log),
        'total_reward': round(env.total_reward, 2),
        'sat_bonus': round(env.sat_bonus, 2),
        'energy': round(env.energy, 2),
        'final_grid': _serialize_state(env),
        'success': env.done and env.dropoff_time is not None,
        'pickup_time': env.pickup_time,
        'dropoff_time': env.dropoff_time,
    })


@app.route('/api/sim/state')
def sim_state():
    """Get current simulation grid state."""
    if sim_env is None:
        return jsonify({'error': 'No active simulation'}), 400
    return jsonify({'ok': True, 'state': _serialize_state(sim_env)})


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
    live_eval = _evaluate_current_policy(n_rides=25) if agent.trained else {}
    return jsonify({
        'ok': True,
        'trained': agent.trained,
        'gcn': gcn_model.summary() if gcn_model else {'trained': False},
        'live_eval': live_eval,
        'hyperparams': {
            'alpha': agent.alpha,
            'gamma': agent.gamma,
            'epsilon': round(agent.epsilon, 4),
            'epsilon_decay': agent.epsilon_decay,
            'epsilon_min': agent.epsilon_min,
        },
        'training_stats': agent.training_stats(),
    })


@app.route('/api/gcn')
def gcn_stats():
    """Return graph model training summary."""
    return jsonify({
        'ok': True,
        'gcn': gcn_model.summary() if gcn_model else {'trained': False},
    })


@app.route('/api/ride-history')
def ride_history():
    """Return recent persisted ride history from MySQL."""
    limit = int(request.args.get('limit', 25))
    limit = max(1, min(limit, 100))
    if not _ensure_ride_history_table():
        return jsonify({
            'ok': False,
            'db_enabled': False,
            'error': db_error or 'MySQL ride history is not configured',
            'rides': [],
        }), 200

    conn = _mysql_connection(include_database=True)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT
            id, ride_id, created_at, mode, pickup_index, pickup_label,
            dropoff_index, dropoff_label, hazard_source, success,
            total_reward, total_steps, energy_remaining, satisfaction_bonus,
            pickup_time, dropoff_time,
            CASE WHEN live_samples_json IS NULL OR live_samples_json = '[]' THEN 0 ELSE 1 END AS has_live_samples,
            CASE WHEN map_details_json IS NULL OR map_details_json = '{}' THEN 0 ELSE 1 END AS has_map_details
        FROM ride_history
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    for row in rows:
        if row.get('created_at') is not None:
            row['created_at'] = row['created_at'].isoformat()
        row['success'] = bool(row.get('success'))
        row['has_live_samples'] = bool(row.get('has_live_samples'))
        row['has_map_details'] = bool(row.get('has_map_details'))

    return jsonify({'ok': True, 'db_enabled': True, 'rides': rows})


@app.route('/api/ride-history/<int:ride_id>')
def ride_history_detail(ride_id):
    """Return one ride with persisted route, step, and live map JSON details."""
    if not _ensure_ride_history_table():
        return jsonify({
            'ok': False,
            'db_enabled': False,
            'error': db_error or 'MySQL ride history is not configured',
        }), 200

    conn = _mysql_connection(include_database=True)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT *
        FROM ride_history
        WHERE id = %s
        LIMIT 1
    """, (ride_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if not row:
        return jsonify({'ok': False, 'error': 'Ride not found'}), 404

    if row.get('created_at') is not None:
        row['created_at'] = row['created_at'].isoformat()
    row['success'] = bool(row.get('success'))
    for key in ('route_json', 'steps_json', 'live_warnings_json', 'live_samples_json', 'map_details_json'):
        value = row.pop(key, None)
        row[key.replace('_json', '')] = json.loads(value) if value else None

    return jsonify({'ok': True, 'db_enabled': True, 'ride': row})


@app.route('/api/live/hazards')
def live_hazards():
    """Fetch live traffic/weather and return normalized hazard grids."""
    try:
        payload = _fetch_live_hazard_payload()
    except RuntimeError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 502

    return jsonify({'ok': True, **payload})


@app.route('/api/benchmark', methods=['POST'])
def benchmark():
    """Run N rides with the trained agent and return summary stats."""
    data = request.get_json(silent=True) or {}
    n_rides = int(data.get('n_rides', 20))
    n_rides = max(1, min(n_rides, 100))
    seed = data.get('seed')
    rng = np.random.RandomState(int(seed)) if seed is not None else np.random.RandomState()

    results = []
    for i in range(n_rides):
        env = TaxiEnvExtended(seed=int(rng.randint(1, 1_000_000)))
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
        'seeded': seed is not None,
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

def _optional_location_index(data, key):
    if key not in data or data.get(key) in (None, ''):
        return None
    value = int(data.get(key))
    if value < 0 or value >= len(LOCS):
        raise ValueError(f"{key} must be between 0 and {len(LOCS) - 1}")
    return value


def _apply_ride_selection(env, data):
    """Apply optional user-selected pickup and drop-off points."""
    pickup = _optional_location_index(data, 'pickup')
    dropoff = _optional_location_index(data, 'dropoff')
    if pickup is None and dropoff is None:
        return
    if pickup is None:
        pickup = env.pass_loc
    if dropoff is None:
        dropoff = env.destination
    if pickup == dropoff:
        raise ValueError('Pickup and drop-off must be different')

    env.pass_loc = pickup
    env.destination = dropoff
    env.pass_on_board = False
    env.done = False
    env.pickup_time = None
    env.dropoff_time = None


def _normalized_grid(value, name):
    """Validate a grid of normalized traffic/weather values."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"{name} must be a {GRID_SIZE}x{GRID_SIZE} array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.clip(arr, 0.0, 1.0)


def _grid_lat_lng(row, col):
    """Match the frontend grid cell to a real Hyderabad coordinate."""
    denominator = max(GRID_SIZE - 1, 1)
    base_lat = MAP_LAT_MAX - (row / denominator) * (MAP_LAT_MAX - MAP_LAT_MIN)
    base_lng = MAP_LNG_MIN + (col / denominator) * (MAP_LNG_MAX - MAP_LNG_MIN)
    lat_offset, lng_offset = (0.0, 0.0)
    if row < len(ROAD_NODE_OFFSETS) and col < len(ROAD_NODE_OFFSETS[row]):
        lat_offset, lng_offset = ROAD_NODE_OFFSETS[row][col]
    lat = base_lat + lat_offset
    lng = base_lng + lng_offset

    for idx, (loc_row, loc_col) in enumerate(LOCS):
        if (loc_row, loc_col) == (row, col):
            loc = MAP_LOCS_ADJUSTED[idx]
            return loc['lat'], loc['lng']

    return lat, lng


def _read_json_url(url, timeout=6):
    """Read a small JSON API response."""
    request_obj = urllib.request.Request(url, headers={'User-Agent': 'taxi-ai-demo/1.0'})
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        body = ''
        try:
            body = exc.read().decode('utf-8', errors='replace')[:300]
        except Exception:
            body = ''
        print(f"External API HTTP error {exc.code} for {url}: {body}", flush=True)
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        print(f"External API network error for {url}: {exc.reason}", flush=True)
        raise RuntimeError(f"Network error: {exc.reason}") from exc
    except TimeoutError as exc:
        print(f"External API timeout for {url}", flush=True)
        raise RuntimeError("Request timed out") from exc


def _fetch_tomtom_traffic(lat, lng, api_key):
    params = urllib.parse.urlencode({
        'key': api_key,
        'point': f'{lat:.6f},{lng:.6f}',
        'unit': 'kmph',
    })
    url = f'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/13/json?{params}'
    data = _read_json_url(url, timeout=2)
    flow = data.get('flowSegmentData', {})
    current = float(flow.get('currentSpeed') or 0.0)
    free = float(flow.get('freeFlowSpeed') or 0.0)
    confidence = float(flow.get('confidence') or 0.0)
    closed = bool(flow.get('roadClosure', False))
    coordinates = (flow.get('coordinates') or {}).get('coordinate') or []
    if isinstance(coordinates, dict):
        coordinates = [coordinates]
    road_points = [
        [float(point['latitude']), float(point['longitude'])]
        for point in coordinates
        if 'latitude' in point and 'longitude' in point
    ]

    if closed:
        return 1.0, {
            'current_speed': current,
            'free_flow_speed': free,
            'confidence': confidence,
            'road_closed': True,
            'road_points': road_points,
        }
    if free <= 0:
        return 0.0, {
            'current_speed': current,
            'free_flow_speed': free,
            'confidence': confidence,
            'road_closed': False,
            'road_points': road_points,
        }

    delay = max(0.0, min(1.0, 1.0 - (current / free)))
    # Keep low-confidence traffic from dominating route choices.
    intensity = min(1.0, delay * (0.55 + confidence * 0.45))
    return intensity, {
        'current_speed': round(current, 1),
        'free_flow_speed': round(free, 1),
        'confidence': round(confidence, 2),
        'road_closed': False,
        'road_points': road_points,
    }


def _fetch_weatherapi(lat, lng, api_key):
    params = urllib.parse.urlencode({
        'key': api_key,
        'q': f'{lat:.6f},{lng:.6f}',
        'aqi': 'no',
    })
    url = f'https://api.weatherapi.com/v1/current.json?{params}'
    data = _read_json_url(url, timeout=4)
    current = data.get('current') or {}
    condition_data = current.get('condition') or {}
    precipitation = float(current.get('precip_mm') or 0.0)
    wind_kph = float(current.get('wind_kph') or 0.0)
    gust_kph = float(current.get('gust_kph') or wind_kph)
    clouds = float(current.get('cloud') or 0.0)
    visibility_km = float(current.get('vis_km') or 10.0)
    humidity = float(current.get('humidity') or 0.0)
    condition = condition_data.get('text', '')

    precipitation_score = min(1.0, precipitation / 12.0)
    wind_score = min(1.0, max(wind_kph, gust_kph) / 65.0)
    cloud_score = min(1.0, clouds / 100.0) * 0.35
    visibility_score = max(0.0, min(1.0, (10.0 - visibility_km) / 10.0))
    humidity_score = min(1.0, humidity / 100.0) * 0.15
    intensity = min(1.0, precipitation_score * 0.55 + wind_score * 0.20 + cloud_score + visibility_score * 0.25 + humidity_score)

    return intensity, {
        'condition': condition,
        'precip_mm': round(precipitation, 2),
        'wind_kph': round(wind_kph, 2),
        'gust_kph': round(gust_kph, 2),
        'clouds_pct': round(clouds, 1),
        'humidity_pct': round(humidity, 1),
        'visibility_km': round(visibility_km, 1),
    }


def _fetch_tomtom_route_points(cells):
    """Fetch TomTom road geometry for a planned grid-cell route."""
    api_key = os.getenv('TOMTOM_API_KEY')
    if not api_key or len(cells) < 2:
        return None

    key = tuple((cell['row'], cell['col']) for cell in cells)
    if key in route_geometry_cache:
        return route_geometry_cache[key]

    locations = []
    for cell in cells:
        lat, lng = _grid_lat_lng(cell['row'], cell['col'])
        locations.append(f'{lat:.6f},{lng:.6f}')
    encoded_locations = urllib.parse.quote(':'.join(locations), safe=':,.')
    params = urllib.parse.urlencode({
        'key': api_key,
        'traffic': 'true',
        'travelMode': 'car',
        'computeTravelTimeFor': 'all',
    })
    url = f'https://api.tomtom.com/routing/1/calculateRoute/{encoded_locations}/json?{params}'
    data = _read_json_url(url, timeout=2)
    route = (data.get('routes') or [{}])[0]
    points = []
    for leg in route.get('legs', []) or []:
        for point in leg.get('points', []) or []:
            if 'latitude' in point and 'longitude' in point:
                next_point = [float(point['latitude']), float(point['longitude'])]
                if not points or points[-1] != next_point:
                    points.append(next_point)

    if len(points) < 2:
        return None

    summary = route.get('summary') or {}
    result = {
        'points': points,
        'summary': {
            'length_m': summary.get('lengthInMeters'),
            'travel_time_s': summary.get('travelTimeInSeconds'),
            'traffic_delay_s': summary.get('trafficDelayInSeconds'),
        }
    }
    if len(route_geometry_cache) > 128:
        route_geometry_cache.clear()
    route_geometry_cache[key] = result
    return result


def _fetch_live_hazard_payload():
    """Build normalized hazard grids from TomTom traffic and WeatherAPI.com."""
    tomtom_key = os.getenv('TOMTOM_API_KEY')
    weather_key = os.getenv('WEATHERAPI_KEY')
    if not tomtom_key or not weather_key:
        missing = []
        if not tomtom_key:
            missing.append('TOMTOM_API_KEY')
        if not weather_key:
            missing.append('WEATHERAPI_KEY')
        raise RuntimeError(f"Missing API key env var(s): {', '.join(missing)}")

    traffic_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    weather_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    samples = []
    center_lat, center_lng = _grid_lat_lng(GRID_SIZE // 2, GRID_SIZE // 2)
    weather_failed = False
    try:
        city_weather, city_weather_meta = _fetch_weatherapi(center_lat, center_lng, weather_key)
    except Exception as exc:
        city_weather = 0.0
        city_weather_meta = {'error': str(exc), 'condition': 'unavailable'}
        weather_failed = True

    traffic_failures = []
    traffic_successes = 0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            lat, lng = _grid_lat_lng(row, col)
            skip_traffic_fetch = traffic_successes == 0 and len(traffic_failures) >= 4
            if skip_traffic_fetch:
                traffic = 0.0
                traffic_meta = {
                    'error': 'TomTom traffic skipped after repeated API failures',
                    'current_speed': None,
                    'free_flow_speed': None,
                    'confidence': 0.0,
                    'road_closed': False,
                    'road_points': [],
                }
            else:
                try:
                    traffic, traffic_meta = _fetch_tomtom_traffic(lat, lng, tomtom_key)
                    traffic_successes += 1
                except Exception as exc:
                    traffic = 0.0
                    traffic_meta = {
                        'error': str(exc),
                        'current_speed': None,
                        'free_flow_speed': None,
                        'confidence': 0.0,
                        'road_closed': False,
                        'road_points': [],
                    }
                    traffic_failures.append({'row': row, 'col': col, 'error': str(exc)})

            traffic_grid[row][col] = traffic
            weather_grid[row][col] = city_weather
            samples.append({
                'row': row,
                'col': col,
                'lat': round(lat, 6),
                'lng': round(lng, 6),
                'traffic': round(float(traffic), 3),
                'weather': round(float(city_weather), 3),
                'traffic_meta': traffic_meta,
                'weather_meta': city_weather_meta,
            })

    source_bits = []
    source_bits.append('tomtom-partial' if traffic_failures else 'tomtom')
    source_bits.append('weatherapi-fallback' if weather_failed else 'weatherapi')
    warnings = traffic_failures[:8]
    if weather_failed:
        warnings = [{'row': None, 'col': None, 'error': city_weather_meta['error']}] + warnings

    return {
        'traffic_grid': traffic_grid.round(3).tolist(),
        'weather_grid': weather_grid.round(3).tolist(),
        'hazard_source': 'live:' + '+'.join(source_bits),
        'live_samples': samples,
        'live_warnings': warnings,
        'fetched_at': int(time.time()),
    }


def _apply_live_sample_metadata(env, data):
    samples = data.get('live_samples')
    if samples:
        env.live_samples = samples
        env.hazard_source = data.get('hazard_source', 'live')
        env.hazards_fetched_at = data.get('fetched_at')
        env.live_warnings = data.get('live_warnings', [])


def _apply_external_hazards(env, data):
    """Apply caller-provided real-world traffic/weather grids when present."""
    has_traffic = 'traffic_grid' in data
    has_weather = 'weather_grid' in data
    if not has_traffic and not has_weather:
        return 'generated'

    if has_traffic:
        env.traffic_grid = _normalized_grid(data['traffic_grid'], 'traffic_grid')
    if has_weather:
        env.weather_grid = _normalized_grid(data['weather_grid'], 'weather_grid')

    _apply_live_sample_metadata(env, data)
    env.hazard_source = data.get('hazard_source', 'external_normalized_grid')
    env.hazards_fetched_at = data.get('fetched_at')
    return data.get('hazard_source', 'external_normalized_grid')


def _train_gcn_model(samples=180, epochs=140, seed=None):
    """Train the graph convolution cost model from experienced rewards."""
    samples = max(20, min(samples, 1000))
    epochs = max(10, min(epochs, 500))
    seed = int(seed) if seed is not None else int(time.time() * 1000) % 1_000_000
    rng = np.random.RandomState(seed)
    model = TaxiGCNCostModel(seed=int(rng.randint(1, 1_000_000)))
    graph_samples = []

    max_steps = 120
    for _ in range(samples):
        env = TaxiEnvExtended(seed=int(rng.randint(1, 1_000_000)))
        for step in range(max_steps):
            if not env.pass_on_board:
                target_row, target_col = LOCS[env.pass_loc]
            else:
                target_row, target_col = LOCS[env.destination]

            before_row, before_col = env.taxi_row, env.taxi_col
            features = model.features_for(env, target_row, target_col)
            action = _experience_collection_action(env, rng, exploration=0.22)
            _, reward, done, info = env.step(action)

            if action < 4 and info.get('event') in ('move', 'energy_depleted'):
                graph_samples.append(model.transition_cost_sample(
                    features,
                    env.taxi_row,
                    env.taxi_col,
                    reward,
                ))
            elif action < 4 and (env.taxi_row, env.taxi_col) == (before_row, before_col):
                graph_samples.append(model.transition_cost_sample(
                    features,
                    before_row,
                    before_col,
                    reward,
                ))

            if done:
                break

    model.fit(graph_samples, epochs=epochs, target_source='episode_reward_transitions')
    return model


def _experience_collection_action(env, rng, exploration=0.2):
    """Collect reward samples with a mix of exploratory and goal-directed moves."""
    if not env.pass_on_board:
        target_row, target_col = LOCS[env.pass_loc]
        if (env.taxi_row, env.taxi_col) == (target_row, target_col):
            return 4
    else:
        target_row, target_col = LOCS[env.destination]
        if (env.taxi_row, env.taxi_col) == (target_row, target_col):
            return 5

    legal_moves = [
        action for action in range(4)
        if env.can_move(env.taxi_row, env.taxi_col, action)[0]
    ]
    if legal_moves and rng.rand() < exploration:
        return int(rng.choice(legal_moves))
    return _formula_path_action(env, target_row, target_col)


def _formula_path_action(env, target_row, target_col):
    """Plan with the baseline formula only, avoiding any already-loaded GCN."""
    path, actions = _astar_path_from(env, (env.taxi_row, env.taxi_col), (target_row, target_col), None)
    if actions:
        return actions[0]
    return _heuristic_action(env)


def _gcn_reward_shaper(model, weight=0.18):
    """Inject learned graph cost into the Bellman target during Q-learning."""
    def shape(env, action, reward, next_state):
        if model is None or not model.trained or action >= 4:
            return reward
        if not env.pass_on_board:
            target_row, target_col = LOCS[env.pass_loc]
        else:
            target_row, target_col = LOCS[env.destination]
        costs = model.predict_costs(env, target_row, target_col)
        row, col = env.taxi_row, env.taxi_col
        return float(reward) - (float(weight) * float(costs[row][col]))

    return shape


def _evaluate_current_policy(n_rides=25, max_steps=200):
    """Run a fresh random evaluation snapshot for dashboard stats."""
    rewards = []
    steps = []
    successes = 0
    rng = np.random.RandomState()

    for _ in range(n_rides):
        env = TaxiEnvExtended(seed=int(rng.randint(1, 1_000_000)))
        for _ in range(max_steps):
            action = _planned_action(env)
            _, _, done, _ = env.step(action)
            if done:
                break
        rewards.append(float(env.total_reward))
        steps.append(int(env.steps))
        if env.done and env.dropoff_time is not None:
            successes += 1

    return {
        'rides': n_rides,
        'success_rate': round(successes / n_rides * 100, 1),
        'avg_reward': round(float(np.mean(rewards)), 2),
        'max_reward': round(float(np.max(rewards)), 2),
        'min_reward': round(float(np.min(rewards)), 2),
        'avg_steps': round(float(np.mean(steps)), 1),
    }


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


def _move_cost(env, row, col, gcn_costs=None):
    """
    Hazard score for entering a cell.
    Used as the A* tie-breaker after minimizing the number of steps.
    """
    if gcn_costs is not None:
        return float(gcn_costs[row][col])

    traffic = float(env.traffic_grid[row][col])
    weather = float(env.weather_grid[row][col])
    combined_risk = max(traffic, weather) + (min(traffic, weather) * 0.65)

    base_cost = traffic * 2.8 + weather * 2.2
    traffic_risk = (traffic ** 2) * 7.5
    weather_risk = (weather ** 2) * 6.0
    combined_penalty = (combined_risk ** 3) * 9.0
    severe_penalty = 0.0

    if traffic >= 0.55:
        severe_penalty += 8.0 + (traffic - 0.55) * 18.0
    if weather >= 0.50:
        severe_penalty += 6.5 + (weather - 0.50) * 16.0
    if traffic >= 0.45 and weather >= 0.40:
        severe_penalty += 12.0 + combined_risk * 4.0
    if traffic >= 0.80 or weather >= 0.75:
        severe_penalty += 20.0

    return base_cost + traffic_risk + weather_risk + combined_penalty + severe_penalty


def _compare_cost(candidate, current):
    return candidate < current


def _astar_path(env, target_row, target_col):
    """
    Use A* to minimize steps first, then choose the safest route among
    equally short paths.
    """
    start = (env.taxi_row, env.taxi_col)
    target = (target_row, target_col)

    gcn_costs = None
    if gcn_model is not None and gcn_model.trained:
        gcn_costs = gcn_model.predict_costs(env, target_row, target_col)

    return _astar_path_from(env, start, target, gcn_costs)


def _astar_path_from(env, start, target, gcn_costs=None):
    """Return an A* route from any start cell using the supplied cost grid."""
    target_row, target_col = target
    if start == target:
        return [start], []

    start_h = abs(target_row - start[0]) + abs(target_col - start[1])
    frontier = [(start_h, 0.0, 0, 0.0, start)]
    best_cost = {start: (0, 0.0)}
    parents = {}

    while frontier:
        _, _, steps_so_far, risk_so_far, node = heapq.heappop(frontier)
        if (steps_so_far, risk_so_far) != best_cost.get(node):
            continue
        if node == target:
            break

        row, col = node
        for action in range(4):
            ok, nr, nc = env.can_move(row, col, action)
            if not ok:
                continue
            next_node = (nr, nc)
            next_steps = steps_so_far + 1
            next_risk = risk_so_far + _move_cost(env, nr, nc, gcn_costs)
            next_cost = (next_steps, next_risk)
            if _compare_cost(next_cost, best_cost.get(next_node, (float('inf'), float('inf')))):
                best_cost[next_node] = next_cost
                parents[next_node] = (node, action)
                heuristic = abs(target_row - nr) + abs(target_col - nc)
                heapq.heappush(frontier, (next_steps + heuristic, next_risk, next_steps, next_risk, next_node))

    if target not in parents:
        return [start], []

    path = [target]
    actions = []
    node = target
    while node != start:
        parent, action = parents[node]
        path.append(parent)
        actions.append(action)
        node = parent
    path.reverse()
    actions.reverse()
    return path, actions


def _shortest_path_action(env, target_row, target_col):
    """Return the next A* action toward the current target."""
    path, actions = _astar_path(env, target_row, target_col)
    if actions:
        return actions[0]
    return _heuristic_action(env)


def _planned_route(env):
    """Return the active A* route in grid coordinates."""
    if not env.pass_on_board:
        target_row, target_col = LOCS[env.pass_loc]
        phase = 'pickup'
    else:
        target_row, target_col = LOCS[env.destination]
        phase = 'dropoff'

    path, _ = _astar_path(env, target_row, target_col)
    cells = [{'row': row, 'col': col} for row, col in path]
    route = {
        'phase': phase,
        'target': {'row': target_row, 'col': target_col},
        'cells': cells,
    }
    if str(getattr(env, 'hazard_source', '')).startswith('live:'):
        try:
            geometry = _fetch_tomtom_route_points(cells)
        except Exception:
            geometry = None
        if geometry:
            route['road_points'] = geometry['points']
            route['road_summary'] = geometry['summary']
    return route


def _cell_conditions(env, row, col):
    """Return traffic and weather values for a grid cell."""
    traffic = float(env.traffic_grid[row][col]) if hasattr(env, 'traffic_grid') else 0.0
    weather = float(env.weather_grid[row][col]) if hasattr(env, 'weather_grid') else 0.0
    return traffic, weather


def _path_risk(env, path, gcn_costs=None):
    """Return the hazard/model cost for entering each cell in a path."""
    return sum(_move_cost(env, row, col, gcn_costs) for row, col in path[1:])


def _route_options(env, target_row, target_col):
    """Score each legal first move by full route length and route risk."""
    target = (target_row, target_col)
    gcn_costs = None
    cost_source = 'GCN learned cost' if gcn_model is not None and gcn_model.trained else 'traffic/weather formula'
    if gcn_model is not None and gcn_model.trained:
        gcn_costs = gcn_model.predict_costs(env, target_row, target_col)

    options = []
    for candidate in range(4):
        ok, row, col = env.can_move(env.taxi_row, env.taxi_col, candidate)
        if not ok:
            continue
        suffix, suffix_actions = _astar_path_from(env, (row, col), target, gcn_costs)
        if not suffix_actions and (row, col) != target:
            continue
        path = [(env.taxi_row, env.taxi_col)] + suffix
        traffic, weather = _cell_conditions(env, row, col)
        distance = abs(target_row - row) + abs(target_col - col)
        options.append({
            'action': candidate,
            'action_name': ACTION_NAMES[candidate],
            'next': {'row': row, 'col': col},
            'steps': len(path) - 1,
            'risk': _path_risk(env, path, gcn_costs),
            'distance': distance,
            'traffic': traffic,
            'weather': weather,
        })

    options.sort(key=lambda item: (item['steps'], item['risk']))
    return options, cost_source


def _shortest_next_actions(env, target_row, target_col):
    """Return legal move actions that reduce Manhattan distance to the target."""
    current_distance = abs(target_row - env.taxi_row) + abs(target_col - env.taxi_col)
    actions = []

    for candidate in range(4):
        ok, row, col = env.can_move(env.taxi_row, env.taxi_col, candidate)
        if not ok:
            continue
        distance = abs(target_row - row) + abs(target_col - col)
        if distance == current_distance - 1:
            actions.append(candidate)

    return actions


def _decision_reason(env, action):
    """Explain why the planner selected the current action."""
    route = _planned_route(env)
    phase = route['phase']
    target = route['target']
    target_label = LOC_LABELS[env.pass_loc] if phase == 'pickup' else LOC_LABELS[env.destination]

    if action == 4:
        return f"taxi is on pickup zone {LOC_LABELS[env.pass_loc]}, so picking up is the required next action"
    if action == 5:
        return f"taxi is on destination zone {LOC_LABELS[env.destination]}, so dropping off completes the ride"

    cells = route.get('cells', [])
    if len(cells) >= 2:
        current = cells[0]
        nxt = cells[1]
        traffic, weather = _cell_conditions(env, nxt['row'], nxt['col'])
        current_distance = abs(target['row'] - current['row']) + abs(target['col'] - current['col'])
        next_distance = abs(target['row'] - nxt['row']) + abs(target['col'] - nxt['col'])
        distance_text = f"distance {current_distance}->{next_distance}"
        if next_distance == current_distance:
            distance_text = f"keeps distance at {next_distance}"
        elif next_distance > current_distance:
            distance_text = f"distance {current_distance}->{next_distance} to avoid a worse cell"

        options, cost_source = _route_options(env, target['row'], target['col'])
        selected = next((item for item in options if item['action'] == action), None)
        ranked = options[:3]
        comparison = ", ".join(
            f"{item['action_name']}={item['steps']} steps/risk {item['risk']:.2f}"
            for item in ranked
        )
        if selected:
            best_steps = options[0]['steps']
            best_risk = options[0]['risk']
            tied_best = [
                item for item in options
                if item['steps'] == best_steps and abs(item['risk'] - best_risk) < 0.01
            ]
            decision_text = (
                f"selected {ACTION_NAMES[action]} because it has the best route score "
                f"({selected['steps']} steps, risk {selected['risk']:.2f})"
            )
            if len(tied_best) > 1 and selected in tied_best:
                tied_names = "/".join(item['action_name'] for item in tied_best)
                decision_text = (
                    f"selected {ACTION_NAMES[action]} because it ties for best route score "
                    f"with {tied_names} ({selected['steps']} steps, risk {selected['risk']:.2f}); "
                    "A* broke the tie by queue order"
                )
            if comparison:
                decision_text += f" vs {comparison}"
        else:
            decision_text = f"selected {ACTION_NAMES[action]} from the planned route"

        return (
            f"{phase} route to {target_label} ({target['row']},{target['col']}): "
            f"move from ({current['row']},{current['col']}) to ({nxt['row']},{nxt['col']}), "
            f"{distance_text}; next cell traffic {traffic:.2f}, weather {weather:.2f}; "
            f"{decision_text}; cost source: {cost_source}"
        )

    return "no better detour was available, so this keeps the ride progressing toward the goal"


def _serialize_state(env):
    state = env.get_grid_state()
    state['planned_route'] = _planned_route(env)
    live_by_cell = {
        (sample.get('row'), sample.get('col')): sample
        for sample in getattr(env, 'live_samples', []) or []
    }
    for cell in state.get('cells', []):
        sample = live_by_cell.get((cell['row'], cell['col']))
        if sample:
            cell['lat'] = sample.get('lat')
            cell['lng'] = sample.get('lng')
            cell['traffic_meta'] = sample.get('traffic_meta', {})
            cell['weather_meta'] = sample.get('weather_meta', {})
        else:
            lat, lng = _grid_lat_lng(cell['row'], cell['col'])
            cell['lat'] = round(lat, 6)
            cell['lng'] = round(lng, 6)
    state['hazard_source'] = getattr(env, 'hazard_source', 'generated')
    state['hazards_fetched_at'] = getattr(env, 'hazards_fetched_at', None)
    return state


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


def _build_log_message(info, action, reason=None):
    from taxi_env import ACTION_NAMES
    parts = [f"{ACTION_NAMES[action]}"]
    if reason:
        parts.append(f"Why: {reason}")
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
#  STARTUP
# ──────────────────────────────────────────────

QTABLE_PATH = os.path.join(os.path.dirname(__file__), 'qtable_trained.npy')
GCN_PATH = os.path.join(os.path.dirname(__file__), 'gcn_trained.npz')

if os.path.exists(GCN_PATH):
    try:
        gcn_model = TaxiGCNCostModel.load(GCN_PATH)
    except Exception:
        gcn_model = None

if __name__ == '__main__':
    print("\nServer ready at http://localhost:5000")
    print("Model starts untrained. Train it from the UI before simulating.\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
