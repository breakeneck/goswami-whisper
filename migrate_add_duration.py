#!/usr/bin/env python3
"""Migration script to add duration_seconds column to contents table."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pymysql

def main():
    host = os.getenv('MYSQL_HOST', 'localhost')
    port = int(os.getenv('MYSQL_PORT', 3306))
    user = os.getenv('MYSQL_USER', 'goswami')
    password = os.getenv('MYSQL_PASSWORD', 'goswamipassword')
    database = os.getenv('MYSQL_DATABASE', 'goswami_whisper')

    print(f"Connecting to MySQL at {host}:{port}...")

    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    cursor = conn.cursor()

    try:
        cursor.execute('ALTER TABLE contents ADD COLUMN duration_seconds FLOAT DEFAULT NULL')
        print('Column duration_seconds added successfully to contents table')
    except pymysql.err.OperationalError as e:
        if 'Duplicate column' in str(e):
            print('Column duration_seconds already exists in contents table')
        else:
            print(f'Error: {e}')
            raise

    conn.commit()

    print("\nCurrent contents table structure:")
    cursor.execute('SHOW COLUMNS FROM contents')
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    print("\nMigration completed successfully!")

if __name__ == '__main__':
    main()

