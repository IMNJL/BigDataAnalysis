"""Simple UI for interacting with MCP stubs and running RAG+Gemini demo.

Run:
  python ui/app.py

The UI assumes Coordinator at http://127.0.0.1:5005 and DataBroker at http://127.0.0.1:5006
"""
from flask import Flask, render_template_string, request, redirect, url_for
import requests
import os
import subprocess

APP = Flask(__name__)
COORD = os.environ.get('COORD_BASE', 'http://127.0.0.1:5005')
DATA = os.environ.get('DATA_BASE', 'http://127.0.0.1:5006')

INDEX_HTML = '''
<html>
<head><title>MCP UI</title></head>
<body>
  <h1>MCP Dashboard</h1>
  <h2>Agents</h2>
  <pre>{{ agents }}</pre>
  <h2>Tasks</h2>
  <pre>{{ tasks }}</pre>
  <h2>Artifacts</h2>
  <ul>
  {% for aid, rec in artifacts.items() %}
    <li>{{aid}} - {{rec.meta.source}} ({{rec.meta.type}}) - <a href="{{data_base}}/artifact/{{aid}}/download">download</a> - <a href="/run_rag?artifact_id={{aid}}">Run RAG+Gemini</a></li>
  {% endfor %}
  </ul>
</body>
</html>
'''


@APP.route('/')
def index():
    agents = requests.get(f"{COORD}/").text
    try:
        resp = requests.get(f"{COORD}/tasks")
        tasks = resp.json()
    except Exception:
        tasks = {}
    try:
        # Try to extract artifacts by scraping DataBroker dashboard
        db_dash = requests.get(f"{DATA}/dashboard").text
        # crude fallback: show dashboard content as one artifact entry
        artifacts = {'dashboard_html': {'meta': {'source':'dashboard','type':'html'}}}
    except Exception:
        artifacts = {}

    return render_template_string(INDEX_HTML, agents=agents, tasks=tasks, artifacts=artifacts, data_base=DATA)


@APP.route('/run_rag')
def run_rag():
    artifact_id = request.args.get('artifact_id')
    if not artifact_id:
        return 'artifact_id required', 400
    # spawn rag demo in background and redirect to index
    cmd = ['python3', os.path.join('..','demo','rag_demo_gemini.py'), '--artifact-id', artifact_id]
    subprocess.Popen(cmd, cwd=os.path.dirname(__file__))
    return redirect(url_for('index'))


if __name__ == '__main__':
    APP.run(port=5010, debug=False)
