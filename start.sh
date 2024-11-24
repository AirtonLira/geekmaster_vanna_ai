#!/bin/bash
docker-compose up -d
echo "Waiting 20 seconds for Ollama to start..."
sleep 20
curl -X POST http://localhost:11434/api/pull -d '{
"name": "llama2:7b"
}'