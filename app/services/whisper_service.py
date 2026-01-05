import os
import subprocess
import threading
import time

import whisper
import yt_dlp
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

        # Speed factors for different models (approximate processing time relative to audio duration)
        # These are calibrated for modern GPU/CPU - actual transcription is very fast
        # For a 70-min file, we want progress to feel smooth even if it only takes 30-60 seconds
        speed_factors = {
            'tiny': 0.005,   # ~20 seconds for 70 min file
            'base': 0.01,    # ~40 seconds for 70 min file
            'small': 0.02,   # ~80 seconds for 70 min file
            'medium': 0.05,  # ~3.5 min for 70 min file
            'large': 0.1     # ~7 min for 70 min file
        }
        speed_factor = speed_factors.get(model_name, 0.01)
        # Minimum estimated duration is 15 seconds to handle short files
        # This gives smoother progress for fast transcriptions
        estimated_duration = max(15.0, duration * speed_factor) if duration > 0 else 30

        # Update initial duration in database
        if transcription_id and app:
            WhisperService._update_progress_in_db(
                app, transcription_id, 0.0,
                current_pos=0.0,
                duration=duration if duration > 0 else None
            )

        # Load local Whisper model
        print(f"Loading local Whisper model: {model_name}")

        # Track model loading time for progress calculation
        model_load_start = time.time()
        model = whisper.load_model(model_name)
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.1f}s")

        # Add model load time to estimated duration
        total_estimated_duration = estimated_duration + model_load_time

        # Progress tracking in background thread
        progress_stop_event = threading.Event()

        def update_progress():
            """Update progress based on estimated time."""
            print(f"Local Whisper progress thread started: transcription_id={transcription_id}, app={app is not None}")
            if not transcription_id or not app:
                print("Progress thread exiting early - no transcription_id or app")
                return
            start_time = time.time()
            last_progress = 0.0
            # Adaptive update interval: faster updates for shorter estimated durations
            update_interval = max(0.3, min(2.0, total_estimated_duration / 50))
            print(f"Progress update interval: {update_interval:.1f}s, estimated duration: {total_estimated_duration:.1f}s")

            while not progress_stop_event.is_set():
                elapsed = time.time() - start_time
                # Linear progress up to 95%, capped
                raw_progress = (elapsed / total_estimated_duration) * 100
                progress = min(95.0, raw_progress)

                # Update if progress increased by at least 1% or 0.5 seconds have passed
                if progress - last_progress >= 1.0:
                    if duration > 0:
                        current_pos = min(duration * 0.95, (progress / 100.0) * duration)
                    else:
                        current_pos = 0.0
                    WhisperService._update_progress_in_db(
                        app, transcription_id, progress,
                        current_pos=current_pos,
                        status_check='transcribing'
                    )
                    last_progress = progress
                time.sleep(update_interval)

        # Start progress tracking thread
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()

        # Transcribe using local Whisper
        try:
            print(f"Starting local Whisper transcription for: {file_path}")
            result = model.transcribe(file_path, verbose=False)
        finally:
            progress_stop_event.set()
            progress_thread.join(timeout=2)

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
