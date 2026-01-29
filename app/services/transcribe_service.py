"""Service for transcribing audio/video files using various providers."""

import os
import subprocess
import time
from typing import Optional, Callable

import whisper
import whisper.transcribe
import yt_dlp
import tqdm


class TqdmProgressTracker:
    """Wrapper to capture tqdm progress from Whisper's internal processing."""

    def __init__(self, callback=None):
        self.callback = callback
        self.current_progress = 0.0
        self._original_tqdm = tqdm.tqdm

    def __call__(self, *args, **kwargs):
        """Create a wrapped tqdm instance."""
        tracker = self

        class ProgressTqdm(tracker._original_tqdm):
            def update(self, n=1):
                result = super().update(n)
                if self.total and self.total > 0:
                    progress = (self.n / self.total) * 100
                    tracker.current_progress = progress
                    if tracker.callback:
                        tracker.callback(progress)
                return result

        return ProgressTqdm(*args, **kwargs)


class TranscribeService:
    """Service for transcribing audio/video files."""

    WHISPER_MODELS = ['medium', 'large', 'large-v3']
    FASTER_WHISPER_MODELS = ['medium', 'large-v2', 'large-v3']

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
    def transcribe(
        file_path: str,
        provider: str,
        model: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        Transcribe an audio/video file.

        Args:
            file_path: Path to the audio/video file
            provider: Provider name (whisper, faster-whisper)
            model: Model name
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            Transcribed text
        """
        if provider == 'whisper':
            return TranscribeService._transcribe_with_whisper(file_path, model, progress_callback)
        elif provider == 'faster-whisper':
            return TranscribeService._transcribe_with_faster_whisper(file_path, model, progress_callback)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def _transcribe_with_whisper(
        file_path: str,
        model_name: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """Transcribe using OpenAI Whisper."""
        # Validate model name
        if model_name not in TranscribeService.WHISPER_MODELS:
            model_name = 'medium'

        # Get audio duration for progress tracking
        duration = TranscribeService.get_audio_duration(file_path)

        # Load local Whisper model
        print(f"Loading local Whisper model: {model_name}")
        model_load_start = time.time()
        model = whisper.load_model(model_name)
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.1f}s")

        # Create progress tracker
        last_progress = [0.0]

        def on_progress(progress):
            if progress - last_progress[0] >= 2.0:
                print(f"[WHISPER PROGRESS] {progress:.1f}%")
                if progress_callback:
                    progress_callback(progress)
                last_progress[0] = progress

        progress_tracker = TqdmProgressTracker(callback=on_progress)

        # Patch tqdm to capture progress
        original_whisper_tqdm = getattr(whisper.transcribe, 'tqdm', None)
        original_tqdm_tqdm = tqdm.tqdm

        try:
            print(f"Starting local Whisper transcription for: {file_path}")
            if original_whisper_tqdm is not None:
                whisper.transcribe.tqdm = progress_tracker
            tqdm.tqdm = progress_tracker

            # Use initial_prompt to guide the model toward Russian transcription
            # and condition_on_previous_text=False to prevent hallucination spreading
            initial_prompt = """Харе Кришна. Это устная лекция на русском языке.
            В речи присутствует большое количество санскритских имён,
            эпитетов и терминов гаудия-вайшнавской традиции.
            Присутствуют имена и названия, связанные с Кришной,
            Радхой, Враджем, преданными, ачарьями, лилами и шастрами.
            Текст передаётся дословно, без художественной обработки.
            """

            result = model.transcribe(
                file_path,
                verbose=False,
                language="ru",
                task="transcribe",
                initial_prompt=initial_prompt,
                condition_on_previous_text=False,
                # temperature=0.0,
                # beam_size=5
            )
        finally:
            if original_whisper_tqdm is not None:
                whisper.transcribe.tqdm = original_whisper_tqdm
            tqdm.tqdm = original_tqdm_tqdm

        if progress_callback:
            progress_callback(100.0)

        print(f"Local Whisper transcription complete")
        return result["text"]

    @staticmethod
    def _transcribe_with_faster_whisper(
        file_path: str,
        model_name: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """Transcribe using Faster Whisper."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")

        # Validate model name
        if model_name not in TranscribeService.FASTER_WHISPER_MODELS:
            model_name = 'medium'

        # Get audio duration for progress tracking
        duration = TranscribeService.get_audio_duration(file_path)

        # Load Faster Whisper model
        print(f"Loading Faster Whisper model: {model_name}")
        model_load_start = time.time()

        # Try to use CUDA if available, fallback to CPU
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
        except Exception:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")

        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.1f}s")

        print(f"Starting Faster Whisper transcription for: {file_path}")

        # Use initial_prompt to guide the model toward Russian transcription
        initial_prompt = "Харе Кришна. Шрила Прабхупада. Преданное служение. Бхагавад-гита. Шримад-Бхагаватам."
        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            language="ru",
            task="transcribe",
            initial_prompt=initial_prompt,
            condition_on_previous_text=False
        )

        # Collect all segments with progress tracking
        text_parts = []
        total_duration = info.duration if hasattr(info, 'duration') else duration

        for segment in segments:
            text_parts.append(segment.text)
            if progress_callback and total_duration > 0:
                progress = (segment.end / total_duration) * 100
                progress_callback(min(progress, 99.0))

        if progress_callback:
            progress_callback(100.0)

        print(f"Faster Whisper transcription complete")
        return " ".join(text_parts)

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
