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
import requests

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
                               transcribe_status, duration, text, transcribe_txt, draft
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

    @staticmethod
    def list_media(page: int = 1, per_page: int = 20,
                   lang_filter: str = None,
                   has_text: bool = None,
                   has_transcribe_txt: bool = None,
                   has_draft: bool = None,
                   id_filter: str = None,
                   title_filter: str = None,
                   status_filter: str = None,
                   order_by: str = 'id',
                   order_dir: str = 'desc') -> dict:
        """
        List all media records from the Goswami database with pagination and filters.

        Args:
            page: Page number (1-based)
            per_page: Items per page
            lang_filter: 'ENG' or 'RUS' - filter by language
            has_text: True/False - filter by text field presence
            has_transcribe_txt: True/False - filter by transcribe_txt field presence
            has_draft: True/False - filter by draft field presence
            id_filter: Exact ID match (string representation of integer)
            title_filter: Partial title match (ILIKE %query%)
            status_filter: transcribe_status value to filter by
            order_by: Column to sort by (id, title, duration, language, text, transcribe_txt, draft, transcribe_status)
            order_dir: 'asc' or 'desc'

        Returns:
            dict with 'items', 'total', 'page', 'per_page', 'pages'
        """
        try:
            with GoswamiDBService._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Build WHERE clause
                    conditions = []
                    params = []

                    # ID filter: exact match
                    if id_filter:
                        try:
                            conditions.append("id = %s")
                            params.append(int(id_filter))
                        except ValueError:
                            pass  # Ignore non-numeric ID filter

                    # Title filter: partial match with ILIKE
                    if title_filter:
                        conditions.append("title ILIKE %s")
                        params.append(f"%{title_filter}%")

                    if lang_filter:
                        conditions.append("language = %s")
                        params.append(lang_filter)

                    if has_text is True:
                        conditions.append("text IS NOT NULL AND text != ''")
                    elif has_text is False:
                        conditions.append("(text IS NULL OR text = '')")

                    if has_transcribe_txt is True:
                        conditions.append("transcribe_txt IS NOT NULL AND transcribe_txt != ''")
                    elif has_transcribe_txt is False:
                        conditions.append("(transcribe_txt IS NULL OR transcribe_txt = '')")

                    if has_draft is True:
                        conditions.append("draft IS NOT NULL AND draft != ''")
                    elif has_draft is False:
                        conditions.append("(draft IS NULL OR draft = '')")

                    # Status filter
                    if status_filter:
                        if status_filter == 'pending':
                            conditions.append("(transcribe_status IS NULL OR transcribe_status = '')")
                        else:
                            conditions.append("transcribe_status = %s")
                            params.append(status_filter)

                    where_clause = " AND ".join(conditions) if conditions else "1=1"

                    # Validate and map order_by
                    allowed_orders = {
                        'id': 'id', 'title': 'title', 'duration': 'duration',
                        'language': 'language', 'text': 'text',
                        'transcribe_txt': 'transcribe_txt', 'draft': 'draft',
                        'transcribe_status': 'transcribe_status',
                        'status': 'transcribe_status'
                    }
                    order_col = allowed_orders.get(order_by, 'id')
                    order_direction = 'ASC' if order_dir.lower() == 'asc' else 'DESC'

                    # For text-like columns, sort nulls/empty last using CASE
                    if order_col in ('text', 'transcribe_txt', 'draft'):
                        if order_direction == 'ASC':
                            # Non-empty first, then alphabetical
                            order_clause = f"CASE WHEN {order_col} IS NULL OR {order_col} = '' THEN 1 ELSE 0 END ASC, {order_col} ASC"
                        else:
                            # Empty first (reverse), then reverse alphabetical
                            order_clause = f"CASE WHEN {order_col} IS NULL OR {order_col} = '' THEN 0 ELSE 1 END ASC, {order_col} DESC"
                    else:
                        order_clause = f"{order_col} {order_direction}"

                    # Count total
                    count_query = f"SELECT COUNT(*) as total FROM media WHERE {where_clause}"
                    cur.execute(count_query, params)
                    total = cur.fetchone()['total']

                    # Fetch page
                    offset = (page - 1) * per_page
                    query = f"""
                        SELECT id, title, duration, language, transcribe_status,
                               text, transcribe_txt, draft, category_id,
                               file_url, occurrence_date
                        FROM media
                        WHERE {where_clause}
                        ORDER BY {order_clause}
                        LIMIT %s OFFSET %s
                    """
                    cur.execute(query, params + [per_page, offset])
                    rows = [dict(row) for row in cur.fetchall()]

                    # Format dates and durations for JSON serialization
                    for row in rows:
                        if row.get('occurrence_date'):
                            row['occurrence_date'] = row['occurrence_date'].isoformat()
                        # Handle timedelta for duration field
                        if row.get('duration') is not None:
                            if hasattr(row['duration'], 'total_seconds'):
                                row['duration'] = row['duration'].total_seconds()
                        # Handle None values from database
                        for key in ['text', 'transcribe_txt', 'draft', 'language', 'transcribe_status']:
                            if key not in row:
                                row[key] = None

                    pages = (total + per_page - 1) // per_page if total > 0 else 1

                    return {
                        'items': rows,
                        'total': total,
                        'page': page,
                        'per_page': per_page,
                        'pages': pages
                    }
        except psycopg2.OperationalError as e:
            logger.error(f"Cannot connect to Goswami database: {e}")
            raise ConnectionError(
                f"Cannot connect to Goswami database. "
                f"Check GOSWAMI_DB_* settings. Error: {e}"
            )
        except Exception as e:
            logger.error(f"Error listing media: {e}")
            raise

    @staticmethod
    def update_transcribe_status(media_id: int, status: str) -> bool:
        """
        Update transcribe_status for a media record.

        Args:
            media_id: The media ID
            status: New status value

        Returns:
            True if updated successfully
        """
        try:
            with GoswamiDBService._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE media SET transcribe_status = %s WHERE id = %s",
                        (status, media_id)
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating transcribe_status for media {media_id}: {e}")
            raise

    @staticmethod
    def update_draft(media_id: int, draft: str) -> bool:
        """
        Update draft field for a media record.

        Args:
            media_id: The media ID
            draft: New draft text

        Returns:
            True if updated successfully
        """
        try:
            with GoswamiDBService._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE media SET draft = %s WHERE id = %s",
                        (draft, media_id)
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating draft for media {media_id}: {e}")
            raise

    @staticmethod
    def claim_media_for_formatting(media_id: int) -> bool:
        """
        Atomically claim a media record for formatting.

        Sets transcribe_status to 'started_formatting' only if it's not already
        in a started state. Returns True if the claim succeeded (row was updated),
        False if another worker already claimed it.

        Args:
            media_id: The media ID to claim

        Returns:
            True if claimed successfully, False if already claimed
        """
        try:
            with GoswamiDBService._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE media
                        SET transcribe_status = 'started_formatting'
                        WHERE id = %s
                          AND transcribe_status NOT IN ('started_formatting', 'started_transcribe')
                    """, (media_id,))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error claiming media {media_id}: {e}")
            return False

    @staticmethod
    def get_pending_formatting_media(limit: int = 100) -> list:
        """
        Get media records that have transcribe_txt but no draft.
        These are candidates for batch formatting.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of media record dicts
        """
        try:
            with GoswamiDBService._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, title, transcribe_txt, duration, language, transcribe_status
                        FROM media
                        WHERE transcribe_txt IS NOT NULL AND transcribe_txt != ''
                          AND (draft IS NULL OR draft = '')
                          AND transcribe_status NOT IN ('started_formatting', 'started_transcribe')
                        ORDER BY id ASC
                        LIMIT %s
                    """, (limit,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching pending formatting media: {e}")
            raise

    @staticmethod
    def get_vllm_models() -> list:
        """
        Get available vLLM models by querying the vLLM API.

        Returns:
            List of model IDs
        """
        try:
            base_url = current_app.config.get('VLLM_BASE_URL', 'http://localhost:8000/v1')
            timeout = current_app.config.get('VLLM_TIMEOUT', 3600)
            resp = requests.get(f"{base_url}/models", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [m['id'] for m in data.get('data', [])]
            return []
        except Exception as e:
            logger.warning(f"Error fetching vLLM models: {e}")
            return []
