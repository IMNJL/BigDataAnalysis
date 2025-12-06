"""Lightweight MCP stub servers: Coordinator and DataBroker

These are simple Flask-based stubs demonstrating the endpoints and message formats.
They are NOT production-ready but useful for local integration testing.
"""
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
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


@COORD.route('/dashboard', methods=['GET'])
def coord_dashboard():
    # Simple HTML dashboard listing agents and tasks
    html = ['<html><head><title>Coordinator MCP</title></head><body>']
    html.append('<h1>Coordinator MCP</h1>')
    html.append('<h2>Registered agents</h2><ul>')
    for aid, info in AGENTS.items():
        html.append(f'<li>{aid}: {info.get("capabilities")}</li>')
    html.append('</ul>')
    html.append('<h2>Tasks</h2><ul>')
    for tid, t in TASKS.items():
        html.append(f'<li>{tid}: {t.get("spec")}</li>')
    html.append('</ul>')
    html.append('</body></html>')
    return '\n'.join(html), 200


@COORD.errorhandler(404)
def coord_not_found(e):
    return jsonify({'error': 'not found', 'service': 'Coordinator MCP', 'available_endpoints': ['/register','/task','/tasks','/notify','/dashboard']}), 404

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
    if id in ARTIFACTS:
        return jsonify(ARTIFACTS[id])
    # Fallback: check /tmp for a file with this id (in case server restarted and lost in-memory index)
    tmp_path = os.path.join('/tmp', id)
    if os.path.exists(tmp_path):
        # return basic metadata pointing to local path
        return jsonify({'meta': {'source': os.path.basename(tmp_path), 'type': 'unknown'}, 'path': tmp_path}), 200
    return jsonify({'error': 'not found'}), 404


@DATA.route('/artifact/<id>/download', methods=['GET'])
def download_artifact(id):
    # Prefer in-memory record
    rec = ARTIFACTS.get(id)
    if rec:
        path = rec.get('path')
        meta = rec.get('meta', {})
    else:
        # Fallback to /tmp/<id>
        path = os.path.join('/tmp', id)
        # Try to construct minimal metadata from the filesystem
        if os.path.exists(path):
            meta = {'source': os.path.basename(path), 'type': 'unknown'}
        else:
            meta = {}

    if not path or not os.path.exists(path):
        return jsonify({'error': 'file missing', 'path': path}), 404

    # derive download filename from metadata if available
    fname = meta.get('source') or os.path.basename(path)
    try:
        # Prefer modern parameter name
        return send_file(path, as_attachment=True, download_name=fname)
    except TypeError:
        # Some Flask versions use `attachment_filename` instead of `download_name`
        try:
            return send_file(path, as_attachment=True, attachment_filename=fname)
        except Exception as e:
            print('send_file fallback failed:', e)
            return jsonify({'error': 'failed to send file', 'detail': str(e)}), 500
    except Exception as e:
        # Log unexpected errors to console for easier debugging
        print('send_file error:', e)
        return jsonify({'error': 'failed to send file', 'detail': str(e)}), 500
    # If send_file failed for any reason above, attempt a manual streaming response as a last resort
    try:
        def generate():
            with open(path, 'rb') as fh:
                while True:
                    chunk = fh.read(8192)
                    if not chunk:
                        break
                    yield chunk

        headers = {
            'Content-Disposition': f'attachment; filename="{fname}"'
        }
        return Response(stream_with_context(generate()), headers=headers, mimetype='application/octet-stream')
    except Exception as e:
        print('manual stream failed:', e)
        return jsonify({'error': 'failed to stream file', 'detail': str(e)}), 500

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


@DATA.route('/dashboard', methods=['GET'])
def data_dashboard():
    html = ['<html><head><title>DataBroker MCP</title></head><body>']
    html.append('<h1>DataBroker MCP</h1>')
    html.append('<h2>Artifacts</h2><ul>')
    for aid, rec in ARTIFACTS.items():
        meta = rec.get('meta', {})
        html.append(f'<li>{aid}: {meta.get("source")} ({meta.get("type")}) - <a href="/artifact/{aid}/download">download</a></li>')
    html.append('</ul>')
    html.append('</body></html>')
    return '\n'.join(html), 200


@DATA.errorhandler(404)
def data_not_found(e):
    return jsonify({'error': 'not found', 'service': 'DataBroker MCP', 'available_endpoints': ['/store','/artifact/<id>','/artifact/<id>/download','/search','/dashboard']}), 404

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
