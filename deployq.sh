git fetch origin main
git reset --hard origin/main

source ./venv/bin/activate

# Flask-Migrate: Initialize if migrations/versions doesn't exist
if [ ! -d "migrations/versions" ]; then
    echo "Initializing Flask-Migrate..."
    flask db init
    flask db migrate -m "Initial migration"
fi

# Run any pending migrations
echo "Running database migrations..."
flask db upgrade || echo "Migration upgrade skipped (tables may already exist)"

python main.py
