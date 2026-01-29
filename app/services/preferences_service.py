"""Service for managing user preferences stored in INI file."""

import configparser
import os


class PreferencesService:
    """Service for saving and loading user preferences to/from INI file."""

    PREFERENCES_FILE = 'preferences.ini'
    SECTION_TRANSCRIBE = 'transcribe'
    SECTION_FORMAT = 'format'

    @classmethod
    def _get_preferences_path(cls):
        """Get the full path to the preferences file."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(base_dir, cls.PREFERENCES_FILE)

    @classmethod
    def _load_config(cls):
        """Load the config parser with existing preferences."""
        config = configparser.ConfigParser()
        path = cls._get_preferences_path()
        if os.path.exists(path):
            config.read(path)
        return config

    @classmethod
    def _save_config(cls, config):
        """Save config to the preferences file."""
        path = cls._get_preferences_path()
        with open(path, 'w') as f:
            config.write(f)

    @classmethod
    def get_transcribe_preferences(cls):
        """Get last used transcribe provider and model."""
        config = cls._load_config()
        if config.has_section(cls.SECTION_TRANSCRIBE):
            return {
                'provider': config.get(cls.SECTION_TRANSCRIBE, 'provider', fallback='whisper'),
                'model': config.get(cls.SECTION_TRANSCRIBE, 'model', fallback='base')
            }
        return {'provider': 'whisper', 'model': 'base'}

    @classmethod
    def set_transcribe_preferences(cls, provider, model):
        """Save transcribe provider and model preferences."""
        config = cls._load_config()
        if not config.has_section(cls.SECTION_TRANSCRIBE):
            config.add_section(cls.SECTION_TRANSCRIBE)
        config.set(cls.SECTION_TRANSCRIBE, 'provider', provider)
        config.set(cls.SECTION_TRANSCRIBE, 'model', model)
        cls._save_config(config)

    @classmethod
    def get_format_preferences(cls):
        """Get last used format provider and model."""
        config = cls._load_config()
        if config.has_section(cls.SECTION_FORMAT):
            return {
                'provider': config.get(cls.SECTION_FORMAT, 'provider', fallback='anthropic'),
                'model': config.get(cls.SECTION_FORMAT, 'model', fallback='')
            }
        return {'provider': 'anthropic', 'model': ''}

    @classmethod
    def set_format_preferences(cls, provider, model):
        """Save format provider and model preferences."""
        config = cls._load_config()
        if not config.has_section(cls.SECTION_FORMAT):
            config.add_section(cls.SECTION_FORMAT)
        config.set(cls.SECTION_FORMAT, 'provider', provider)
        config.set(cls.SECTION_FORMAT, 'model', model)
        cls._save_config(config)

    @classmethod
    def get_all_preferences(cls):
        """Get all preferences."""
        return {
            'transcribe': cls.get_transcribe_preferences(),
            'format': cls.get_format_preferences()
        }
