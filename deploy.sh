git fetch origin main
git reset --hard origin/main

./venv/bin/pip install -r requirements.txt
docker compose up -d
sleep 5  # Wait for MySQL to be ready

# Initialize migrations if not exists
if [ ! -d "migrations" ]; then
    ./venv/bin/flask db init
    ./venv/bin/flask db migrate -m "Initial migration"
fi
./venv/bin/flask db upgrade

./venv/bin/python main.py
