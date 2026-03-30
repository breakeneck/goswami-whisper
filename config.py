import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    # Database
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_PORT = os.environ.get('MYSQL_PORT', '3306')
    MYSQL_USER = os.environ.get('MYSQL_USER', 'goswami')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'goswamipassword')
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'goswami_whisper')

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@"
        f"{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File uploads
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(os.path.dirname(__file__), 'uploads'))
    ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'webm', 'm4a', 'ogg', 'flac', 'mpeg', 'mpga'}
    MAX_CONTENT_LENGTH = 4 * 1024 * 1024 * 1024  # 4 GB max file size

    # OpenAI
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

    # Anthropic (Claude)
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

    # Google Gemini
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

    # xAI (Grok)
    XAI_API_KEY = os.environ.get('XAI_API_KEY', '')

    # Zhipu AI (智谱AI/BigModel)
    ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY', '')

    # Hugging Face
    HF_TOKEN = os.environ.get('HF_TOKEN', os.environ.get('HUGGINGFACE_HUB_TOKEN', ''))
    QWEN3_ASR_MODEL = os.environ.get('QWEN3_ASR_MODEL', 'Qwen/Qwen3-ASR-1.7B')

    # LM Studio
    LMSTUDIO_BASE_URL = os.environ.get('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
    LMSTUDIO_TIMEOUT = int(os.environ.get('LMSTUDIO_TIMEOUT', '3600'))  # 1 hour default (LM Studio can be very slow)

    # Transcription providers and models
    TRANSCRIBE_PROVIDERS = {
        'whisper': {
            'name': 'OpenAI Whisper',
            'models': ['whisper-1', 'medium', 'large', 'large-v3']
        },
        'faster-whisper': {
            'name': 'Faster Whisper',
            'models': ['medium', 'large-v2', 'large-v3', 'large-v3-turbo']
        },
        'qwen3-asr': {
            'name': 'Qwen3-ASR',
            'models': [QWEN3_ASR_MODEL]
        }
    }

    # Formatting providers and models
    FORMAT_PROVIDERS = {
        'openai': {
            'name': 'OpenAI',
            'models': ['gpt-5.2', 'gpt-5.1', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o-mini']
        },
        'anthropic': {
            'name': 'Anthropic (Claude)',
            'models': [
                'claude-opus-4-6', 'claude-opus-4-5-20251101', 'claude-opus-4-1-20250805', 'claude-opus-4-20250514',
                'claude-sonnet-4-6', 'claude-sonnet-4-5-20250929', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219',
                'claude-haiku-4-5-20251001', 'claude-3-5-haiku-20241022', 'claude-3-haiku-20240307'
            ]
        },
        'gemini': {
            'name': 'Google Gemini',
            'models': ['gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash']
        },
        'xai': {
            'name': 'xAI (Grok)',
            'models': ['grok-3', 'grok-3-fast', 'grok-3-mini', 'grok-3-mini-fast', 'grok-2-1212', 'grok-2-vision-1212', 'grok-vision-beta', 'grok-beta']
        },
        'zhipu': {
            'name': 'Zhipu AI (智谱)',
            'models': ['glm-4.7', 'glm-4.7-flash', 'glm-4.7-flashx', 'glm-4.6', 'glm-4.6v-flashx', 'glm-4.5-air', 'glm-4-plus', 'glm-4-long', 'glm-4-air', 'glm-4-airx', 'glm-4-flash', 'glm-4-flashx', 'glm-z1-air', 'glm-z1-airx', 'glm-z1-flash']
        },
        'lmstudio': {
            'name': 'LM Studio (Local)',
            'models': []  # Will be loaded dynamically
        }
    }

    # Claude models pricing (per 1M tokens)
    # Format: 'model_name': {name, input, output}
    CLAUDE_MODELS = {
        'claude-opus-4-6': {'name': 'Claude Opus 4.6', 'input': 5.00, 'output': 10.00},
        'claude-opus-4-5-20251101': {'name': 'Claude Opus 4.5', 'input': 5.00, 'output': 10.00},
        'claude-opus-4-1-20250805': {'name': 'Claude Opus 4.1', 'input': 15.00, 'output': 30.00},
        'claude-opus-4-20250514': {'name': 'Claude Opus 4', 'input': 15.00, 'output': 30.00},
        'claude-sonnet-4-6': {'name': 'Claude Sonnet 4.6', 'input': 3.00, 'output': 6.00},
        'claude-sonnet-4-5-20250929': {'name': 'Claude Sonnet 4.5', 'input': 3.00, 'output': 6.00},
        'claude-sonnet-4-20250514': {'name': 'Claude Sonnet 4', 'input': 3.00, 'output': 6.00},
        'claude-3-7-sonnet-20250219': {'name': 'Claude Sonnet 3.7 (retired)', 'input': 3.00, 'output': 6.00},
        'claude-haiku-4-5-20251001': {'name': 'Claude Haiku 4.5', 'input': 1.00, 'output': 2.00},
        'claude-3-5-haiku-20241022': {'name': 'Claude Haiku 3.5 (retired)', 'input': 0.80, 'output': 1.60},
        'claude-3-haiku-20240307': {'name': 'Claude Haiku 3', 'input': 0.25, 'output': 0.50}
    }

    # ChromaDB
    CHROMA_PERSIST_DIR = os.environ.get('CHROMA_PERSIST_DIR', os.path.join(os.path.dirname(__file__), 'chroma_db'))

    # Goswami external PostgreSQL database (for upload by ID)
    GOSWAMI_DB_NAME = os.environ.get('GOSWAMI_DB_NAME', 'goswami.ru')
    GOSWAMI_DB_USER = os.environ.get('GOSWAMI_DB_USER', 'postgres')
    GOSWAMI_DB_PASSWORD = os.environ.get('GOSWAMI_DB_PASSWORD', 'postgres')
    GOSWAMI_DB_HOST = os.environ.get('GOSWAMI_DB_HOST', 'localhost')
    GOSWAMI_DB_PORT = os.environ.get('GOSWAMI_DB_PORT', '5431')

    # Media files root prefix (path to audio files on disk)
    MEDIA_ROOT_PREFIX = os.environ.get('MEDIA_ROOT_PREFIX', '~/hdd/media/bvgm.su')
