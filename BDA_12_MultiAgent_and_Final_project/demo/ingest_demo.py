"""Minimal ingestion demo that registers an agent and posts a CSV to DataBroker.

Usage:
    python ingest_demo.py --csv <path-to-csv>
"""
import argparse
import requests
import os

COORD = 'http://localhost:5005'
DATA = 'http://localhost:5006'


def register_agent(name):
    r = requests.post(f"{COORD}/register", json={'agent_id': name, 'capabilities': ['ingest']})
    return r.json()


def post_csv(path):
    files = {'file': open(path, 'rb')}
    data = {'type': 'csv', 'source': os.path.basename(path)}
    r = requests.post(f"{DATA}/store", files=files, data=data)
    return r.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()

    print('Registering demo agent...')
    print(register_agent('IngestDemoAgent'))
    print('Posting CSV to DataBroker...')
    print(post_csv(args.csv))

if __name__ == '__main__':
    main()
