git fetch origin main
git reset --hard origin/main

docker compose up -d
./venv/bin/python main.py
