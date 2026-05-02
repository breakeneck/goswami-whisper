"""Service for managing user preferences stored in INI file."""

import configparser
import json
import os


class PreferencesService:
    """Service for saving and loading user preferences to/from INI file."""

    PREFERENCES_FILE = 'preferences.ini'
    FORMAT_SAMPLING_FILE = 'format_sampling.json'
    SECTION_TRANSCRIBE = 'transcribe'
    SECTION_FORMAT = 'format'
    SECTION_PROMPT = 'prompt'

    @classmethod
    def _project_base_dir(cls):
        return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    @classmethod
    def _get_preferences_path(cls):
        """Get the full path to the preferences file."""
        return os.path.join(cls._project_base_dir(), cls.PREFERENCES_FILE)

    @classmethod
    def _get_format_sampling_path(cls):
        return os.path.join(cls._project_base_dir(), cls.FORMAT_SAMPLING_FILE)

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
    def load_format_sampling_store(cls) -> dict:
        """Nested dict: provider -> model_id -> sampling params."""
        path = cls._get_format_sampling_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @classmethod
    def save_format_sampling_store(cls, store: dict):
        path = cls._get_format_sampling_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(store, f, indent=2, sort_keys=True)

    @classmethod
    def get_format_sampling_for_model(cls, provider: str, model: str) -> dict:
        if not provider or not model:
            return {}
        return dict(cls.load_format_sampling_store().get(provider, {}).get(model, {}) or {})

    @classmethod
    def sanitize_llm_sampling_params(cls, params: dict) -> dict:
        """Validate/normalize sampling fields from UI or API."""
        if not params or not isinstance(params, dict):
            return {}

        def _clamp(v, lo, hi):
            try:
                x = float(v)
                return max(lo, min(hi, x))
            except (TypeError, ValueError):
                return None

        out = {}

        t = params.get('temperature')
        if t is not None and t != '':
            c = _clamp(t, 0.0, 2.0)
            if c is not None:
                out['temperature'] = c

        tp = params.get('top_p')
        if tp is not None and tp != '':
            c = _clamp(tp, 0.0, 1.0)
            if c is not None:
                out['top_p'] = c

        tk = params.get('top_k')
        if tk is not None and tk != '':
            try:
                k = int(tk)
                if 1 <= k <= 128:
                    out['top_k'] = k
            except (TypeError, ValueError):
                pass

        fp = params.get('frequency_penalty')
        if fp is not None and fp != '':
            c = _clamp(fp, -2.0, 2.0)
            if c is not None:
                out['frequency_penalty'] = c

        pp = params.get('presence_penalty')
        if pp is not None and pp != '':
            c = _clamp(pp, -2.0, 2.0)
            if c is not None:
                out['presence_penalty'] = c

        mt = params.get('max_tokens')
        if mt is not None:
            try:
                n = int(mt)
                if 64 <= n <= 262144:
                    out['max_tokens'] = n
            except (TypeError, ValueError):
                pass

        return out

    @classmethod
    def merge_format_sampling_into_store(cls, provider: str, model: str, params: dict):
        """Upsert sanitized sampling settings for provider+model. None values remove optional keys."""
        if not provider or not model:
            return
        if params is None:
            return

        store = cls.load_format_sampling_store()
        bucket = store.setdefault(provider, {})
        current = dict(bucket.get(model) or {})

        if 'max_tokens' in params and params['max_tokens'] is None:
            current.pop('max_tokens', None)
        if 'top_k' in params and params['top_k'] is None:
            current.pop('top_k', None)

        cleaned = cls.sanitize_llm_sampling_params(params)
        current.update(cleaned)
        bucket[model] = current
        cls.save_format_sampling_store(store)

    @classmethod
    def get_all_preferences(cls):
        """Get all preferences."""
        return {
            'transcribe': cls.get_transcribe_preferences(),
            'format': cls.get_format_preferences(),
            'format_sampling': cls.load_format_sampling_store(),
        }

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

