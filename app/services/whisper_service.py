import os
import subprocess
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
    def transcribe_file(file_path: str, model_name: str = None, transcription_id: int = None) -> str:
        """
        Transcribe an audio/video file using Whisper.

        Args:
            file_path: Path to the audio/video file
            model_name: Whisper model to use (tiny, base, small, medium, large)
            transcription_id: Optional transcription ID for progress tracking

        Returns:
            Transcribed text
        """
        try:
            import whisper
            import threading
            import time
            from flask import current_app
            from app import db
            from app.models.transcription import Transcription

            # Use provided model or fall back to config default
            if model_name is None:
                model_name = current_app.config.get('WHISPER_MODEL', 'base')

            # Validate model name
            if model_name not in WhisperService.VALID_MODELS:
                model_name = 'base'

            # Get audio duration for progress tracking
            duration = WhisperService.get_audio_duration(file_path)

            # Speed factors for different models (approximate real-time factor)
            # e.g., 0.1 means the model processes audio 10x faster than real-time
            speed_factors = {
                'tiny': 0.05,
                'base': 0.1,
                'small': 0.2,
                'medium': 0.4,
                'large': 0.8
            }
            speed_factor = speed_factors.get(model_name, 0.2)
            estimated_duration = duration * speed_factor if duration > 0 else 60

            if transcription_id and duration > 0:
                transcription = Transcription.query.get(transcription_id)
                if transcription:
                    transcription.duration_seconds = duration
                    transcription.progress = 0.0
                    db.session.commit()

            # Load Whisper model
            model = whisper.load_model(model_name)

            # Progress tracking in background thread
            progress_stop_event = threading.Event()

            def update_progress():
                """Update progress based on estimated time."""
                if not transcription_id or duration <= 0:
                    return

                start_time = time.time()
                while not progress_stop_event.is_set():
                    elapsed = time.time() - start_time
                    # Estimate progress based on elapsed time vs estimated duration
                    # Cap at 95% to leave room for actual completion
                    progress = min(95.0, (elapsed / estimated_duration) * 100)

                    # Also estimate current position in audio
                    current_pos = min(duration * 0.95, (elapsed / estimated_duration) * duration)

                    try:
                        transcription = Transcription.query.get(transcription_id)
                        if transcription and transcription.status == 'transcribing':
                            transcription.progress = progress
                            transcription.current_position = current_pos
                            db.session.commit()
                    except Exception:
                        pass

                    time.sleep(1)  # Update every second

            # Start progress tracking thread
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()

            # Transcribe
            try:
                result = model.transcribe(file_path, verbose=False)
            finally:
                # Stop progress tracking
                progress_stop_event.set()
                progress_thread.join(timeout=2)

            # Final progress update based on actual segments
            if transcription_id and duration > 0:
                transcription = Transcription.query.get(transcription_id)
                if transcription:
                    transcription.progress = 100.0
                    transcription.current_position = duration
                    db.session.commit()

            return result["text"]
        except ImportError:
            # Fallback to OpenAI Whisper API if local model not available
            from openai import OpenAI
            from flask import current_app

            client = OpenAI(api_key=current_app.config.get('OPENAI_API_KEY'))

            with open(file_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            return transcript.text

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

        # Configure yt-dlp options
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
            # Get the filename
            filename = ydl.prepare_filename(info)
            # Change extension to mp3 (due to postprocessor)
            base_name = os.path.splitext(filename)[0]
            mp3_path = base_name + '.mp3'

            if os.path.exists(mp3_path):
                return mp3_path
            elif os.path.exists(filename):
                return filename
            else:
                # Find the most recently created file in the output dir
                files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
                if files:
                    return max(files, key=os.path.getctime)
                raise FileNotFoundError(f"Could not find downloaded file for {url}")

