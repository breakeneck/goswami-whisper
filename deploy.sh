sudo -u http git fetch origin main
sudo -u http git reset --hard origin/main

docker compose up -d
./venv/bin/python main.py
