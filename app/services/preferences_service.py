"""Service for managing user preferences stored in INI file."""

import configparser
import os


class PreferencesService:
    """Service for saving and loading user preferences to/from INI file."""

    PREFERENCES_FILE = 'preferences.ini'
    SECTION_TRANSCRIBE = 'transcribe'
    SECTION_FORMAT = 'format'
    SECTION_PROMPT = 'prompt'
    SECTION_LLM_PARAMS = 'llm_params'

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
            'format': cls.get_format_preferences(),
            'llm_params': cls.get_llm_params()
        }

    @classmethod
    def get_llm_params(cls):
        """Get saved LLM generation parameters (temperature, top_p, presence_penalty, frequency_penalty)."""
        config = cls._load_config()
        params = {}
        if config.has_section(cls.SECTION_LLM_PARAMS):
            if config.has_option(cls.SECTION_LLM_PARAMS, 'temperature'):
                params['temperature'] = config.getfloat(cls.SECTION_LLM_PARAMS, 'temperature')
            if config.has_option(cls.SECTION_LLM_PARAMS, 'top_p'):
                params['top_p'] = config.getfloat(cls.SECTION_LLM_PARAMS, 'top_p')
            if config.has_option(cls.SECTION_LLM_PARAMS, 'presence_penalty'):
                params['presence_penalty'] = config.getfloat(cls.SECTION_LLM_PARAMS, 'presence_penalty')
            if config.has_option(cls.SECTION_LLM_PARAMS, 'frequency_penalty'):
                params['frequency_penalty'] = config.getfloat(cls.SECTION_LLM_PARAMS, 'frequency_penalty')
        return params

    @classmethod
    def set_llm_params(cls, temperature=None, top_p=None, presence_penalty=None, frequency_penalty=None):
        """Save LLM generation parameters. Pass None to remove a specific parameter."""
        config = cls._load_config()
        if not config.has_section(cls.SECTION_LLM_PARAMS):
            config.add_section(cls.SECTION_LLM_PARAMS)
        
        if temperature is not None:
            config.set(cls.SECTION_LLM_PARAMS, 'temperature', str(temperature))
        elif config.has_option(cls.SECTION_LLM_PARAMS, 'temperature'):
            config.remove_option(cls.SECTION_LLM_PARAMS, 'temperature')
            
        if top_p is not None:
            config.set(cls.SECTION_LLM_PARAMS, 'top_p', str(top_p))
        elif config.has_option(cls.SECTION_LLM_PARAMS, 'top_p'):
            config.remove_option(cls.SECTION_LLM_PARAMS, 'top_p')
            
        if presence_penalty is not None:
            config.set(cls.SECTION_LLM_PARAMS, 'presence_penalty', str(presence_penalty))
        elif config.has_option(cls.SECTION_LLM_PARAMS, 'presence_penalty'):
            config.remove_option(cls.SECTION_LLM_PARAMS, 'presence_penalty')
            
        if frequency_penalty is not None:
            config.set(cls.SECTION_LLM_PARAMS, 'frequency_penalty', str(frequency_penalty))
        elif config.has_option(cls.SECTION_LLM_PARAMS, 'frequency_penalty'):
            config.remove_option(cls.SECTION_LLM_PARAMS, 'frequency_penalty')
        
        cls._save_config(config)

    @classmethod
    def clear_llm_params(cls):
        """Clear all saved LLM generation parameters (revert to defaults)."""
        config = cls._load_config()
        if config.has_section(cls.SECTION_LLM_PARAMS):
            config.remove_section(cls.SECTION_LLM_PARAMS)
            cls._save_config(config)

    @classmethod
    def get_system_prompt(cls):
        """Get custom system prompt for formatting."""
        config = cls._load_config()
        if config.has_section(cls.SECTION_PROMPT):
            return config.get(cls.SECTION_PROMPT, 'system_prompt', fallback=None)
        return None

    @classmethod
    def set_system_prompt(cls, prompt):
        """Save custom system prompt for formatting."""
        config = cls._load_config()
        if not config.has_section(cls.SECTION_PROMPT):
            config.add_section(cls.SECTION_PROMPT)
        config.set(cls.SECTION_PROMPT, 'system_prompt', prompt)
        cls._save_config(config)

    @classmethod
    def clear_system_prompt(cls):
        """Clear custom system prompt (revert to default)."""
        config = cls._load_config()
        if config.has_section(cls.SECTION_PROMPT):
            config.remove_section(cls.SECTION_PROMPT)
            cls._save_config(config)

    @classmethod
    def get_prompt_history(cls):
        """Get prompt history list."""
        config = cls._load_config()
        history = []
        if config.has_section(cls.SECTION_PROMPT):
            # Read history items (stored as numbered keys)
            import json
            history_json = config.get(cls.SECTION_PROMPT, 'history', fallback='[]')
            try:
                history = json.loads(history_json)
            except:
                history = []
        return history

    @classmethod
    def _save_prompt_history(cls, history):
        """Save prompt history list."""
        import json
        config = cls._load_config()
        if not config.has_section(cls.SECTION_PROMPT):
            config.add_section(cls.SECTION_PROMPT)
        config.set(cls.SECTION_PROMPT, 'history', json.dumps(history))
        cls._save_config(config)

    @classmethod
    def add_prompt_to_history(cls, prompt: str, name: str = None):
        """Add current prompt to history before updating."""
        if not prompt:
            return
        import json
        from datetime import datetime
        history = cls.get_prompt_history()
        # Create entry with timestamp
        entry = {
            'prompt': prompt,
            'name': name or f"Prompt {len(history) + 1}",
            'timestamp': datetime.now().isoformat()
        }
        # Add to beginning of list
        history.insert(0, entry)
        # Keep only last 20 entries
        history = history[:20]
        cls._save_prompt_history(history)

    @classmethod
    def get_prompt_from_history(cls, index: int):
        """Get prompt from history by index."""
        history = cls.get_prompt_history()
        if 0 <= index < len(history):
            return history[index]['prompt']
        return None

    @classmethod
    def delete_prompt_from_history(cls, index: int):
        """Delete prompt from history by index."""
        history = cls.get_prompt_history()
        if 0 <= index < len(history):
            history.pop(index)
            cls._save_prompt_history(history)
            return True
        return False

