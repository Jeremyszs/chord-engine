#!/bin/bash
BASE="http://127.0.0.1:8001"
echo "=== Step 1: Health check ==="
curl -s "$BASE/api/v1/health" | python -m json.tool
echo ""
echo "=== Step 2: Upload demo_chords.wav ==="
RESP=$(curl -s -X POST "$BASE/api/v1/jobs" -F "audio=@samples/demo_chords.wav")
echo "$RESP" | python -m json.tool
JOB_ID=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job ID: $JOB_ID"
echo ""
echo "=== Step 3: Poll status (waiting 10s...) ==="
sleep 10
curl -s "$BASE/api/v1/jobs/$JOB_ID/status" | python -m json.tool
echo ""
echo "=== Step 4: Fetch result ==="
curl -s "$BASE/api/v1/jobs/$JOB_ID/result" | python -m json.tool
