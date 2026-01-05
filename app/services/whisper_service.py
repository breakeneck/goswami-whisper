import os
import subprocess
import tempfile
import yt_dlp


class WhisperService:
    """Service for transcribing audio/video files using Whisper."""

    VALID_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

    @staticmethod
    def transcribe_file(file_path: str, model_name: str = None) -> str:
        """
        Transcribe an audio/video file using Whisper.

        Args:
            file_path: Path to the audio/video file
            model_name: Whisper model to use (tiny, base, small, medium, large)

        Returns:
            Transcribed text
        """
        try:
            import whisper
            from flask import current_app

            # Use provided model or fall back to config default
            if model_name is None:
                model_name = current_app.config.get('WHISPER_MODEL', 'base')

            # Validate model name
            if model_name not in WhisperService.VALID_MODELS:
                model_name = 'base'

            # Load Whisper model
            model = whisper.load_model(model_name)

            # Transcribe
            result = model.transcribe(file_path)

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

