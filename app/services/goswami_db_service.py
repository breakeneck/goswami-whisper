"""
Service for accessing the external Goswami PostgreSQL database.
Used for uploading audio files by media ID from the goswami.ru database.

File path building logic:
    audio_path = MEDIA_ROOT_PREFIX / year / month / file_url
    where:
        - MEDIA_ROOT_PREFIX = env var (e.g., ~/hdd/media/bvgm.su)
        - year = str(occurrence_date.year)
        - month = f"{occurrence_date.month:02d}"
        - file_url = from the media record in the database
"""

import os
import logging
from typing import Optional, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor
from flask import current_app

logger = logging.getLogger(__name__)


class GoswamiDBService:
    """Service for accessing the external Goswami PostgreSQL database."""

    @staticmethod
    def _get_connection():
        """Create a connection to the Goswami database."""
        return psycopg2.connect(
            dbname=current_app.config.get('GOSWAMI_DB_NAME', 'goswami.ru'),
            user=current_app.config.get('GOSWAMI_DB_USER', 'postgres'),
            password=current_app.config.get('GOSWAMI_DB_PASSWORD', 'postgres'),
            host=current_app.config.get('GOSWAMI_DB_HOST', 'localhost'),
            port=current_app.config.get('GOSWAMI_DB_PORT', '5431'),
        )

    @staticmethod
    def _get_root_prefix() -> str:
        """Get the expanded media root prefix path."""
        prefix = current_app.config.get('MEDIA_ROOT_PREFIX', '~/hdd/media/bvgm.su')
        return os.path.expanduser(prefix)

    @staticmethod
    def build_file_path(record: Dict[str, Any]) -> str:
        """
        Build the full file path from a media record.

        Path format: MEDIA_ROOT_PREFIX / year / month / file_url

        Args:
            record: dict with 'file_url' and 'occurrence_date' keys

        Returns:
            Full path to the audio file on disk
        """
        root_prefix = GoswamiDBService._get_root_prefix()
        occurrence_date = record['occurrence_date']
        year_folder = str(occurrence_date.year)
        month_folder = f"{occurrence_date.month:02d}"
        file_url = record['file_url']

        return os.path.join(root_prefix, year_folder, month_folder, file_url)

    @staticmethod
    def get_media_by_id(media_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a media record by ID from the Goswami database.

        Args:
            media_id: The media ID to look up

        Returns:
            dict with media record fields, or None if not found
        """
        try:
            with GoswamiDBService._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, title, file_url, occurrence_date, language,
                               transcribe_status, duration, text
                        FROM media
                        WHERE id = %s
                    """, (media_id,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        except psycopg2.OperationalError as e:
            logger.error(f"Cannot connect to Goswami database: {e}")
            raise ConnectionError(
                f"Cannot connect to Goswami database. "
                f"Check GOSWAMI_DB_* and MEDIA_ROOT_PREFIX settings. Error: {e}"
            )
        except Exception as e:
            logger.error(f"Error fetching media {media_id}: {e}")
            raise

    @staticmethod
    def get_media_file_path(media_id: int) -> Optional[str]:
        """
        Get the full file path for a media record by ID.

        Args:
            media_id: The media ID to look up

        Returns:
            Full path to the audio file, or None if not found
        """
        record = GoswamiDBService.get_media_by_id(media_id)
        if not record:
            return None

        if not record.get('file_url'):
            raise ValueError(f"Media {media_id} has no file_url")

        return GoswamiDBService.build_file_path(record)

    @staticmethod
    def search_media(query: str, limit: int = 20) -> list:
        """
        Search media records by title or ID.

        Args:
            query: Search query (title substring or numeric ID)
            limit: Maximum number of results

        Returns:
            List of media record dicts
        """
        try:
            with GoswamiDBService._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # If query is a number, also search by ID
                    if query.strip().isdigit():
                        cur.execute("""
                            SELECT id, title, file_url, occurrence_date, language,
                                   transcribe_status, duration
                            FROM media
                            WHERE type = 'audio'
                              AND (title ILIKE %s OR id = %s)
                            ORDER BY occurrence_date DESC
                            LIMIT %s
                        """, (f"%{query}%", int(query.strip()), limit))
                    else:
                        cur.execute("""
                            SELECT id, title, file_url, occurrence_date, language,
                                   transcribe_status, duration
                            FROM media
                            WHERE type = 'audio'
                              AND title ILIKE %s
                            ORDER BY occurrence_date DESC
                            LIMIT %s
                        """, (f"%{query}%", limit))

                    return [dict(row) for row in cur.fetchall()]
        except psycopg2.OperationalError as e:
            logger.error(f"Cannot connect to Goswami database: {e}")
            raise ConnectionError(
                f"Cannot connect to Goswami database. "
                f"Check GOSWAMI_DB_* settings. Error: {e}"
            )
        except Exception as e:
            logger.error(f"Error searching media: {e}")
            raise
