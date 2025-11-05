#!/usr/bin/env bash
set -e

# Resolve script dir and project root (two levels up from src/scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"

# Usage: ./run.sh <user_id> [--simulate-llm]
USER_ID=${1:-50}
SIMULATE_FLAG=${2:---simulate-llm}

python3 "${PROJECT_ROOT}/run/recommend_user.py" --user "${USER_ID}" ${SIMULATE_FLAG}
python3 "${PROJECT_ROOT}/make_report.py" --user "${USER_ID}"
echo "Report generated: ${PROJECT_ROOT}/BDA_9_Recommendation_report.docx"