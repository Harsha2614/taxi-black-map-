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
import numpy as np
from flask import Flask, jsonify, request, render_template, Response, stream_with_context

from taxi_env import TaxiEnvExtended, QLearningAgent, LOC_LABELS, LOC_COLORS, LOCS, NUM_STATES, ACTION_NAMES
from gcn_model import TaxiGCNCostModel

app = Flask(__name__)

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

# Hyderabad sample points matching the frontend's 5x5 map grid.
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
]


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
            if live:
                data = {**data, **_fetch_live_hazard_payload()}
            hazard_source = _apply_external_hazards(sim_env, data)
        except ValueError as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 502
    grid = _serialize_state(sim_env)
    return jsonify({
        'ok': True,
        'seed': seed,
        'hazard_source': hazard_source,
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
        grid = _serialize_state(sim_env)

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
        if live:
            data = {**data, **_fetch_live_hazard_payload()}
        hazard_source = _apply_external_hazards(env, data)
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


@app.route('/api/live/hazards')
def live_hazards():
    """Fetch live traffic/weather and return normalized 5x5 hazard grids."""
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

def _normalized_grid(value, name):
    """Validate a 5x5 grid of normalized traffic/weather values."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (5, 5):
        raise ValueError(f"{name} must be a 5x5 array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.clip(arr, 0.0, 1.0)


def _grid_lat_lng(row, col):
    """Match the frontend grid cell to a real Hyderabad coordinate."""
    base_lat = MAP_LAT_MAX - (row / 4) * (MAP_LAT_MAX - MAP_LAT_MIN)
    base_lng = MAP_LNG_MIN + (col / 4) * (MAP_LNG_MAX - MAP_LNG_MIN)
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
    with urllib.request.urlopen(request_obj, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def _fetch_tomtom_traffic(lat, lng, api_key):
    params = urllib.parse.urlencode({
        'key': api_key,
        'point': f'{lat:.6f},{lng:.6f}',
        'unit': 'kmph',
    })
    url = f'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/13/json?{params}'
    data = _read_json_url(url)
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


def _fetch_openweather(lat, lng, api_key):
    params = urllib.parse.urlencode({
        'lat': f'{lat:.6f}',
        'lon': f'{lng:.6f}',
        'appid': api_key,
        'units': 'metric',
    })
    url = f'https://api.openweathermap.org/data/2.5/weather?{params}'
    data = _read_json_url(url)
    rain = float((data.get('rain') or {}).get('1h') or (data.get('rain') or {}).get('3h') or 0.0)
    snow = float((data.get('snow') or {}).get('1h') or (data.get('snow') or {}).get('3h') or 0.0)
    wind = float((data.get('wind') or {}).get('speed') or 0.0)
    clouds = float((data.get('clouds') or {}).get('all') or 0.0)
    visibility = float(data.get('visibility') or 10000.0)
    condition = (data.get('weather') or [{}])[0].get('main', '')

    precipitation_score = min(1.0, (rain + snow) / 12.0)
    wind_score = min(1.0, wind / 18.0)
    cloud_score = min(1.0, clouds / 100.0) * 0.35
    visibility_score = max(0.0, min(1.0, (10000.0 - visibility) / 10000.0))
    intensity = min(1.0, precipitation_score * 0.55 + wind_score * 0.20 + cloud_score + visibility_score * 0.25)

    return intensity, {
        'condition': condition,
        'rain_mm': round(rain, 2),
        'snow_mm': round(snow, 2),
        'wind_mps': round(wind, 2),
        'clouds_pct': round(clouds, 1),
        'visibility_m': int(visibility),
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
    data = _read_json_url(url, timeout=8)
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
    """Build normalized hazard grids from TomTom traffic and OpenWeather."""
    tomtom_key = os.getenv('TOMTOM_API_KEY')
    weather_key = os.getenv('OPENWEATHER_API_KEY')
    if not tomtom_key or not weather_key:
        missing = []
        if not tomtom_key:
            missing.append('TOMTOM_API_KEY')
        if not weather_key:
            missing.append('OPENWEATHER_API_KEY')
        raise RuntimeError(f"Missing API key env var(s): {', '.join(missing)}")

    traffic_grid = np.zeros((5, 5), dtype=float)
    weather_grid = np.zeros((5, 5), dtype=float)
    samples = []
    center_lat, center_lng = _grid_lat_lng(2, 2)
    try:
        city_weather, city_weather_meta = _fetch_openweather(center_lat, center_lng, weather_key)
    except Exception as exc:
        raise RuntimeError(f"OpenWeather fetch failed: {exc}") from exc

    for row in range(5):
        for col in range(5):
            lat, lng = _grid_lat_lng(row, col)
            try:
                traffic, traffic_meta = _fetch_tomtom_traffic(lat, lng, tomtom_key)
            except Exception as exc:
                raise RuntimeError(f"TomTom traffic fetch failed at cell ({row},{col}): {exc}") from exc

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

    return {
        'traffic_grid': traffic_grid.round(3).tolist(),
        'weather_grid': weather_grid.round(3).tolist(),
        'hazard_source': 'live:tomtom+openweather',
        'live_samples': samples,
        'fetched_at': int(time.time()),
    }


def _apply_live_sample_metadata(env, data):
    samples = data.get('live_samples')
    if samples:
        env.live_samples = samples
        env.hazard_source = data.get('hazard_source', 'live')
        env.hazards_fetched_at = data.get('fetched_at')


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
