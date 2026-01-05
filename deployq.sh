git fetch origin main
git reset --hard origin/main

source ./venv/bin/activate

flask db upgrade
python main.py
