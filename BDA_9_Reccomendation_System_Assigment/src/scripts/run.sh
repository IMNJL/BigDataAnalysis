#!/usr/bin/env bash
set -e

# Usage: ./run.sh <user_id> [--simulate-llm]
USER_ID=${1:-50}
SIMULATE=${2:---simulate-llm}

python3 run/recommend_user.py --user ${USER_ID} ${SIMULATE}
python3 make_report.py --user ${USER_ID}
echo "Report generated: BDA_9_Recommendation_report.docx"
# ...existing code...