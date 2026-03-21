#!/bin/bash

set -e

git fetch origin main
git reset --hard origin/main

source ./venv/bin/activate

pip install -r requirements.txt
docker compose up -d
sleep 5  # Wait for MySQL to be ready

# Run any pending migrations
echo "Running database migrations..."
flask db upgrade || {
    echo "Migration upgrade failed, stamping current state..."
    flask db stamp head
}

python main.py
