git fetch origin main
git reset --hard origin/main

./venv/bin/pip install -r requirements.txt
docker compose up -d
sleep 5  # Wait for MySQL to be ready
./venv/bin/flask db upgrade
./venv/bin/python main.py
