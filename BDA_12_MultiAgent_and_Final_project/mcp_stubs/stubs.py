"""Lightweight MCP stub servers: Coordinator and DataBroker

These are simple Flask-based stubs demonstrating the endpoints and message formats.
They are NOT production-ready but useful for local integration testing.
"""
from flask import Flask, request, jsonify
import uuid
import os

COORD = Flask('coordinator')
DATA = Flask('databroker')

# In-memory stores
AGENTS = {}
TASKS = {}
ARTIFACTS = {}

@COORD.route('/register', methods=['POST'])
def register():
    data = request.json
    agent_id = data.get('agent_id') or str(uuid.uuid4())
    AGENTS[agent_id] = data
    return jsonify({'agent_id': agent_id}), 201

@COORD.route('/task', methods=['POST'])
def submit_task():
    data = request.json
    task_id = data.get('task_id') or str(uuid.uuid4())
    TASKS[task_id] = data
    return jsonify({'task_id': task_id}), 201

@COORD.route('/tasks', methods=['GET'])
def list_tasks():
    return jsonify(TASKS)


@COORD.route('/', methods=['GET'])
def coord_root():
    return jsonify({'service': 'Coordinator MCP', 'status': 'running', 'endpoints': ['/register','/task','/tasks','/notify']}), 200


@COORD.route('/favicon.ico')
def coord_favicon():
    return ('', 204)

@COORD.route('/notify', methods=['POST'])
def notify():
    data = request.json
    # naive pass-through
    return jsonify({'ok': True}), 200

@DATA.route('/store', methods=['POST'])
def store():
    data = request.files.get('file')
    meta = request.form.to_dict()
    artifact_id = str(uuid.uuid4())
    path = os.path.join('/tmp', artifact_id)
    if data:
        data.save(path)
    ARTIFACTS[artifact_id] = {'meta': meta, 'path': path}
    return jsonify({'artifact_id': artifact_id}), 201

@DATA.route('/artifact/<id>', methods=['GET'])
def fetch(id):
    if id not in ARTIFACTS:
        return jsonify({'error': 'not found'}), 404
    return jsonify(ARTIFACTS[id])

@DATA.route('/search', methods=['GET'])
def search():
    q = request.args.get('q')
    results = {k:v for k,v in ARTIFACTS.items() if q in str(v.get('meta'))}
    return jsonify(results)


@DATA.route('/', methods=['GET'])
def data_root():
    return jsonify({'service': 'DataBroker MCP', 'status': 'running', 'endpoints': ['/store','/artifact/<id>','/search']}), 200


@DATA.route('/favicon.ico')
def data_favicon():
    return ('', 204)

if __name__ == '__main__':
    # Run both Flask apps for demo using threads to avoid multiprocessing
    # spawn/pickle issues on some platforms (macOS, Windows).
    import threading

    t1 = threading.Thread(target=lambda: COORD.run(port=5005, debug=False, use_reloader=False), daemon=True)
    t2 = threading.Thread(target=lambda: DATA.run(port=5006, debug=False, use_reloader=False), daemon=True)

    t1.start()
    t2.start()

    try:
        # Keep main thread alive while servers run
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print('\nShutting down MCP stub servers...')
