#!/bin/sh
# Serve the model and ask it something over HTTP.
tinyjev serve TinyJev-0.6B --port 8077 &
sleep 20
curl -s localhost:8077/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice. Please fix this ASAP.",
  "questions": {
    "billing": {"type": "noul",   "instructions": "Is this about billing?"},
    "urgency": {"type": "score",  "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today"]}
  }}' | python3 -m json.tool
kill %1
