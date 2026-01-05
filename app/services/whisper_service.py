import os
import subprocess
import time
from unittest.mock import patch

import whisper
import yt_dlp
import tqdm.std


class TqdmProgressTracker:
    """Wrapper to capture tqdm progress from Whisper's internal processing."""

    def __init__(self, callback=None):
        self.callback = callback
        self.current_progress = 0.0
        self._original_tqdm = tqdm.std.tqdm

    def create_wrapper(self):
        """Create a tqdm wrapper class that reports progress."""
        tracker = self
        original_tqdm = self._original_tqdm

        class ProgressTqdm(original_tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                result = super().update(n)
                if self.total and self.total > 0:
                    progress = (self.n / self.total) * 100
                    tracker.current_progress = progress
                    if tracker.callback:
                        tracker.callback(progress)
                return result

        return ProgressTqdm
class WhisperService:
    """Service for transcribing audio/video files using Whisper."""
    VALID_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Get audio duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0
    @staticmethod
    def _update_progress_in_db(app, transcription_id: int, progress: float, current_pos: float = None,
                                duration: float = None, status_check: str = None):
        """
        Update progress in database using raw SQL to avoid session conflicts.
        Args:
            app: Flask app instance
            transcription_id: ID of the transcription
            progress: Progress percentage (0-100)
            current_pos: Current position in seconds (optional)
            duration: Total duration in seconds (optional)
            status_check: Only update if status matches this value (optional)
        """
        from sqlalchemy import text
        from app import db

        def do_update():
            # Use a direct connection to avoid session conflicts
            with db.engine.connect() as conn:
                if status_check:
                    if current_pos is not None and duration is not None:
                        conn.execute(
                            text("UPDATE transcriptions SET progress = :progress, current_position = :pos, duration_seconds = :dur WHERE id = :id AND status = :status"),
                            {"progress": progress, "pos": current_pos, "dur": duration, "id": transcription_id, "status": status_check}
                        )
                    elif current_pos is not None:
                        conn.execute(
                            text("UPDATE transcriptions SET progress = :progress, current_position = :pos WHERE id = :id AND status = :status"),
                            {"progress": progress, "pos": current_pos, "id": transcription_id, "status": status_check}
                        )
                    else:
                        conn.execute(
                            text("UPDATE transcriptions SET progress = :progress WHERE id = :id AND status = :status"),
                            {"progress": progress, "id": transcription_id, "status": status_check}
                        )
                else:
                    if current_pos is not None and duration is not None:
                        conn.execute(
                            text("UPDATE transcriptions SET progress = :progress, current_position = :pos, duration_seconds = :dur WHERE id = :id"),
                            {"progress": progress, "pos": current_pos, "dur": duration, "id": transcription_id}
                        )
                    elif current_pos is not None:
                        conn.execute(
                            text("UPDATE transcriptions SET progress = :progress, current_position = :pos WHERE id = :id"),
                            {"progress": progress, "pos": current_pos, "id": transcription_id}
                        )
                    else:
                        conn.execute(
                            text("UPDATE transcriptions SET progress = :progress WHERE id = :id"),
                            {"progress": progress, "id": transcription_id}
                        )
                conn.commit()

        try:
            # Always push a new app context since threads don't share context
            with app.app_context():
                do_update()
                print(f"Progress update: id={transcription_id}, progress={progress:.1f}%, status_check={status_check}")
        except Exception as e:
            import logging
            logging.error(f"Progress update failed: {e}")
            print(f"Progress update FAILED: {e}")
    @staticmethod
    def compress_audio_for_api(file_path: str, max_size_mb: int = 24) -> str:
        """
        Compress audio file to fit within API size limits.
        Args:
            file_path: Path to the audio file
            max_size_mb: Maximum file size in MB
        Returns:
            Path to the compressed file (or original if already small enough)
        """
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size <= max_size_bytes:
            return file_path
        # Calculate target bitrate based on audio duration
        duration = WhisperService.get_audio_duration(file_path)
        if duration <= 0:
            duration = 600  # Assume 10 minutes if unknown
        # Target bitrate in kbps (with some margin)
        target_bitrate = int((max_size_bytes * 8) / duration / 1000 * 0.9)
        target_bitrate = max(32, min(target_bitrate, 128))  # Clamp between 32-128 kbps
        # Create compressed file
        base_name = os.path.splitext(file_path)[0]
        compressed_path = f"{base_name}_compressed.mp3"
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', file_path,
                '-codec:a', 'libmp3lame', '-b:a', f'{target_bitrate}k',
                '-ac', '1',  # Mono
                '-ar', '16000',  # 16kHz sample rate (sufficient for speech)
                compressed_path
            ], capture_output=True, check=True)
            # Verify the compressed file is small enough
            if os.path.exists(compressed_path) and os.path.getsize(compressed_path) <= max_size_bytes:
                return compressed_path
            else:
                # If still too big, try more aggressive compression
                subprocess.run([
                    'ffmpeg', '-y', '-i', file_path,
                    '-codec:a', 'libmp3lame', '-b:a', '32k',
                    '-ac', '1', '-ar', '16000',
                    compressed_path
                ], capture_output=True, check=True)
                if os.path.exists(compressed_path):
                    return compressed_path
        except Exception:
            pass
        return file_path  # Return original if compression fails
    @staticmethod
    def transcribe_file(file_path: str, model_name: str = None, transcription_id: int = None, app=None) -> str:
        """
        Transcribe an audio/video file using local Whisper.
        Args:
            file_path: Path to the audio/video file
            model_name: Whisper model to use (tiny, base, small, medium, large)
            transcription_id: Optional transcription ID for progress tracking
            app: Flask app instance for context in background threads
        Returns:
            Transcribed text
        """
        from flask import current_app

        # Get app reference for thread context
        if app is None:
            try:
                app = current_app._get_current_object()
            except RuntimeError:
                app = None

        # Use provided model or fall back to config default
        if model_name is None:
            if app:
                model_name = app.config.get('WHISPER_MODEL', 'base')
            else:
                model_name = 'base'

        # Validate model name
        if model_name not in WhisperService.VALID_MODELS:
            model_name = 'base'

        # Get audio duration for progress tracking
        duration = WhisperService.get_audio_duration(file_path)

        # Update initial duration in database
        if transcription_id and app:
            WhisperService._update_progress_in_db(
                app, transcription_id, 0.0,
                current_pos=0.0,
                duration=duration if duration > 0 else None
            )

        # Load local Whisper model
        print(f"Loading local Whisper model: {model_name}")

        model_load_start = time.time()
        model = whisper.load_model(model_name)
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.1f}s")

        # Create progress tracker that captures real tqdm progress from Whisper
        last_db_progress = [0.0]  # Use list to allow modification in nested function

        def on_progress(progress):
            """Callback when tqdm progress updates."""
            # Only update DB when progress changes by at least 2%
            if progress - last_db_progress[0] >= 2.0:
                if transcription_id and app:
                    current_pos = (progress / 100.0) * duration if duration > 0 else 0.0
                    WhisperService._update_progress_in_db(
                        app, transcription_id, progress,
                        current_pos=current_pos,
                        status_check='transcribing'
                    )
                    last_db_progress[0] = progress

        progress_tracker = TqdmProgressTracker(callback=on_progress)

        # Transcribe using local Whisper with progress tracking
        try:
            print(f"Starting local Whisper transcription for: {file_path}")
            # Patch tqdm to capture Whisper's internal progress
            with patch.object(tqdm.std, 'tqdm', progress_tracker.create_wrapper()):
                result = model.transcribe(file_path, verbose=False)
        except Exception as e:
            print(f"Transcription error: {e}")
            raise

        # Final progress update
        if transcription_id and app:
            WhisperService._update_progress_in_db(
                app, transcription_id, 100.0,
                current_pos=duration if duration > 0 else None
            )

        print(f"Local Whisper transcription complete")
        return result["text"]
    @staticmethod
    def download_from_url(url: str, output_dir: str) -> str:
        """
        Download audio from a URL (supports YouTube and other platforms via yt-dlp).
        Args:
            url: URL to download from
            output_dir: Directory to save the downloaded file
        Returns:
            Path to the downloaded file
        """
        os.makedirs(output_dir, exist_ok=True)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            base_name = os.path.splitext(filename)[0]
            mp3_path = base_name + '.mp3'
            if os.path.exists(mp3_path):
                return mp3_path
            elif os.path.exists(filename):
                return filename
            else:
                files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
                if files:
                    return max(files, key=os.path.getctime)
                raise FileNotFoundError(f"Could not find downloaded file for {url}")
