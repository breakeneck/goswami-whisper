"""Service for formatting transcribed text using various LLM providers."""

from openai import OpenAI
from anthropic import Anthropic
from flask import current_app
import requests

# Cache for model context lengths to avoid repeated API calls
_model_context_cache = {}


class FormatService:
    """Service for formatting text using LLM models."""

    SYSTEM_PROMPT = """You are a helpful assistant that formats transcriptions for readability while preserving the original content. Output only the formatted transcription, nothing else."""

    FORMAT_PROMPT = """Please format the following transcription for readability:
1. Add proper punctuation and capitalization
2. Break into logical paragraphs
3. Preserve all the original content and meaning
4. If there are Sanskrit/Hindi terms, keep them as-is
5. Don't add any commentary or summaries

Transcription:
"""

    @staticmethod
    def get_openai_client() -> OpenAI:
        """Get OpenAI client with API key from config."""
        return OpenAI(api_key=current_app.config.get('OPENAI_API_KEY'))

    @staticmethod
    def get_anthropic_client() -> Anthropic:
        """Get Anthropic client with API key from config."""
        return Anthropic(api_key=current_app.config.get('ANTHROPIC_API_KEY'))

    @staticmethod
    def get_lmstudio_client() -> OpenAI:
        """Get LM Studio client (OpenAI-compatible)."""
        return OpenAI(
            base_url=current_app.config.get('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1'),
            api_key='not-needed'
        )

    @staticmethod
    def get_xai_client() -> OpenAI:
        """Get xAI (Grok) client (OpenAI-compatible API)."""
        return OpenAI(
            base_url='https://api.x.ai/v1',
            api_key=current_app.config.get('XAI_API_KEY')
        )

    @staticmethod
    def get_zhipu_client() -> OpenAI:
        """Get Zhipu AI (智谱AI/BigModel) client (OpenAI-compatible API)."""
        return OpenAI(
            base_url='https://open.bigmodel.cn/api/paas/v4',
            api_key=current_app.config.get('ZHIPU_API_KEY')
        )

    @staticmethod
    def get_model_context_length(provider: str, model: str) -> int:
        """
        Fetch the real context length for a model from its API.

        Args:
            provider: Provider name (openai, anthropic, gemini, lmstudio)
            model: Model identifier

        Returns:
            Context length in tokens, or a safe default if unavailable
        """
        cache_key = f"{provider}:{model}"
        if cache_key in _model_context_cache:
            return _model_context_cache[cache_key]

        context_length = None

        try:
            if provider == 'openai':
                context_length = FormatService._get_openai_context_length(model)
            elif provider == 'anthropic':
                context_length = FormatService._get_anthropic_context_length(model)
            elif provider == 'gemini':
                context_length = FormatService._get_gemini_context_length(model)
            elif provider == 'xai':
                context_length = FormatService._get_xai_context_length(model)
            elif provider == 'zhipu':
                context_length = FormatService._get_zhipu_context_length(model)
            elif provider == 'lmstudio':
                context_length = FormatService._get_lmstudio_context_length(model)
        except Exception as e:
            current_app.logger.warning(f"Failed to fetch context length for {provider}:{model}: {e}")

        # Use safe defaults if API fetch failed
        if not context_length:
            context_length = FormatService._get_default_context_length(provider, model)

        _model_context_cache[cache_key] = context_length
        return context_length

    @staticmethod
    def _get_openai_context_length(model: str) -> int:
        """Fetch context length from OpenAI API."""
        try:
            client = FormatService.get_openai_client()
            model_info = client.models.retrieve(model)
            # OpenAI returns context_window in model info for newer models
            # For chat models, check for context_window attribute
            if hasattr(model_info, 'context_window'):
                return model_info.context_window
            # Fallback: known OpenAI model context lengths
            return FormatService._get_openai_known_context(model)
        except Exception:
            return FormatService._get_openai_known_context(model)

    @staticmethod
    def _get_openai_known_context(model: str) -> int:
        """Known context lengths for OpenAI models."""
        model_lower = model.lower()
        # GPT-4o and variants
        if 'gpt-4o' in model_lower:
            return 128000
        # GPT-4 Turbo
        if 'gpt-4-turbo' in model_lower or 'gpt-4-1106' in model_lower or 'gpt-4-0125' in model_lower:
            return 128000
        # GPT-4 32k
        if 'gpt-4-32k' in model_lower:
            return 32768
        # GPT-4
        if 'gpt-4' in model_lower:
            return 8192
        # GPT-3.5 Turbo 16k
        if 'gpt-3.5-turbo-16k' in model_lower:
            return 16384
        # GPT-3.5 Turbo
        if 'gpt-3.5' in model_lower:
            return 4096
        # o1 and o1-mini models
        if 'o1' in model_lower:
            return 128000
        return 8192  # Safe default

    @staticmethod
    def _get_anthropic_context_length(model: str) -> int:
        """Get context length for Anthropic models."""
        model_lower = model.lower()
        # Claude 3.5 and Claude 3 models have 200K context
        if 'claude-3' in model_lower or 'claude-sonnet' in model_lower or 'claude-opus' in model_lower or 'claude-haiku' in model_lower:
            return 200000
        # Claude 2 models
        if 'claude-2' in model_lower:
            return 100000
        return 100000  # Safe default for Claude

    @staticmethod
    def _get_gemini_context_length(model: str) -> int:
        """Fetch context length from Gemini API."""
        try:
            api_key = current_app.config.get('GEMINI_API_KEY')
            if not api_key:
                return FormatService._get_gemini_known_context(model)

            # Use the models.get endpoint to fetch model info
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}?key={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Gemini returns inputTokenLimit and outputTokenLimit
                input_limit = data.get('inputTokenLimit', 0)
                if input_limit > 0:
                    return input_limit
        except Exception:
            pass
        return FormatService._get_gemini_known_context(model)

    @staticmethod
    def _get_gemini_known_context(model: str) -> int:
        """Known context lengths for Gemini models."""
        model_lower = model.lower()
        if 'gemini-1.5-pro' in model_lower:
            return 1000000  # 1M tokens
        if 'gemini-1.5-flash' in model_lower:
            return 1000000  # 1M tokens
        if 'gemini-pro' in model_lower:
            return 32768
        return 32768  # Safe default

    @staticmethod
    def _get_xai_context_length(model: str) -> int:
        """Get context length for xAI (Grok) models."""
        model_lower = model.lower()
        # Grok-3 models have 131072 context
        if 'grok-3' in model_lower:
            return 131072
        # Grok-2 models have 131072 context
        if 'grok-2' in model_lower:
            return 131072
        # Grok beta models
        if 'grok-beta' in model_lower or 'grok-vision-beta' in model_lower:
            return 131072
        return 131072  # Default for Grok models

    @staticmethod
    def _get_zhipu_context_length(model: str) -> int:
        """Get context length for Zhipu AI (智谱AI) models."""
        model_lower = model.lower()
        # GLM-4-Long has 1M context
        if 'glm-4-long' in model_lower:
            return 1000000
        # GLM-4.7 models have 128K context
        if 'glm-4.7-flashx' in model_lower:
            return 128000
        if 'glm-4.7-flash' in model_lower:
            return 128000
        if 'glm-4.7' in model_lower:
            return 128000
        # GLM-4.6 models have 128K context
        if 'glm-4.6v-flashx' in model_lower:
            return 128000
        if 'glm-4.6' in model_lower:
            return 128000
        # GLM-4.5-Air has 128K context
        if 'glm-4.5-air' in model_lower:
            return 128000
        # GLM-Z1 models have 128K context
        if 'glm-z1' in model_lower:
            return 128000
        # GLM-4-Plus has 128K context
        if 'glm-4-plus' in model_lower:
            return 128000
        # GLM-4-Air/AirX have 128K context
        if 'glm-4-air' in model_lower:
            return 128000
        # GLM-4-Flash/FlashX have 128K context
        if 'glm-4-flash' in model_lower:
            return 128000
        # GLM-4 base has 128K context
        if 'glm-4' in model_lower:
            return 128000
        return 128000  # Default for GLM models

    @staticmethod
    def _get_lmstudio_context_length(model: str) -> int:
        """Fetch context length from LM Studio API."""
        try:
            base_url = current_app.config.get('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
            response = requests.get(f"{base_url}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                for model_info in data.get('data', []):
                    if model_info.get('id') == model:
                        ctx_len = model_info.get('context_length')
                        if ctx_len and ctx_len > 0:
                            return ctx_len
        except Exception:
            pass
        # Fall back to inference from model name
        return FormatService.infer_context_length_from_model_name(model)

    @staticmethod
    def _get_default_context_length(provider: str, model: str) -> int:
        """Get a safe default context length for a provider."""
        if provider == 'openai':
            return FormatService._get_openai_known_context(model)
        elif provider == 'anthropic':
            return FormatService._get_anthropic_context_length(model)
        elif provider == 'gemini':
            return FormatService._get_gemini_known_context(model)
        elif provider == 'xai':
            return FormatService._get_xai_context_length(model)
        elif provider == 'zhipu':
            return FormatService._get_zhipu_context_length(model)
        elif provider == 'lmstudio':
            return FormatService.infer_context_length_from_model_name(model)
        return 4096  # Conservative default

    @staticmethod
    def get_max_output_tokens(provider: str, model: str, context_length: int = None) -> int:
        """
        Calculate appropriate max_tokens for output based on model limits.

        Args:
            provider: Provider name
            model: Model identifier
            context_length: Optional pre-fetched context length

        Returns:
            Appropriate max_tokens value for the model
        """
        if not context_length:
            context_length = FormatService.get_model_context_length(provider, model)

        # Different providers have different output limits
        if provider == 'openai':
            # GPT-4o has 16K output limit, others vary
            if 'gpt-4o' in model.lower():
                return 16384
            return min(16384, context_length // 4)
        elif provider == 'anthropic':
            # Claude has up to 8K output tokens by default, can be higher
            return min(16384, context_length // 4)
        elif provider == 'gemini':
            # Gemini 1.5 has 8K output by default, 1.5 Pro has up to 8K
            return min(8192, context_length // 4)
        elif provider == 'xai':
            # Grok models have up to 16K output tokens
            return min(16384, context_length // 4)
        elif provider == 'zhipu':
            # Zhipu GLM-4 models have up to 4K output tokens
            return min(4096, context_length // 4)
        elif provider == 'lmstudio':
            # Local models - be conservative
            return min(16384, context_length // 4)
        return 4096

    @staticmethod
    def format_text(raw_text: str, provider: str, model: str, stream_callback=None, context_length=None) -> str:
        """
        Format raw transcription text using the specified provider and model.

        Args:
            raw_text: Raw transcription text
            provider: Provider name (openai, anthropic, gemini, lmstudio)
            model: Model name
            stream_callback: Optional callback for streaming responses
            context_length: Optional context window size (for LM Studio)

        Returns:
            Formatted text
        """
        if not raw_text:
            return ""

        if provider == 'openai':
            return FormatService._format_with_openai(raw_text, model, stream_callback)
        elif provider == 'anthropic':
            return FormatService._format_with_anthropic(raw_text, model, stream_callback)
        elif provider == 'gemini':
            return FormatService._format_with_gemini(raw_text, model, stream_callback)
        elif provider == 'xai':
            return FormatService._format_with_xai(raw_text, model, stream_callback)
        elif provider == 'zhipu':
            return FormatService._format_with_zhipu(raw_text, model, stream_callback)
        elif provider == 'lmstudio':
            return FormatService._format_with_lmstudio(raw_text, model, stream_callback, context_length)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def _format_with_openai(raw_text: str, model: str, stream_callback=None) -> str:
        """Format text using OpenAI GPT."""
        client = FormatService.get_openai_client()
        max_tokens = FormatService.get_max_output_tokens('openai', model)

        if stream_callback:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FormatService.SYSTEM_PROMPT},
                    {"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}
                ],
                temperature=0.3,
                max_completion_tokens=max_tokens,
                stream=True
            )
            result = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    result += text
                    stream_callback(text)
            return result
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FormatService.SYSTEM_PROMPT},
                    {"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}
                ],
                temperature=0.3,
                max_completion_tokens=max_tokens
            )
            return response.choices[0].message.content

    @staticmethod
    def _format_with_anthropic(raw_text: str, model: str, stream_callback=None) -> str:
        """Format text using Claude (Anthropic)."""
        client = FormatService.get_anthropic_client()
        max_tokens = FormatService.get_max_output_tokens('anthropic', model)

        if stream_callback:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}],
                system=FormatService.SYSTEM_PROMPT
            ) as stream:
                result = ""
                for text in stream.text_stream:
                    result += text
                    stream_callback(text)
                return result
        else:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}],
                system=FormatService.SYSTEM_PROMPT
            )
            return response.content[0].text

    @staticmethod
    def _format_with_gemini(raw_text: str, model: str, stream_callback=None) -> str:
        """Format text using Google Gemini."""
        api_key = current_app.config.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")

        max_tokens = FormatService.get_max_output_tokens('gemini', model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        payload = {
            "contents": [{
                "parts": [{"text": FormatService.SYSTEM_PROMPT + "\n\n" + FormatService.FORMAT_PROMPT + raw_text}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": max_tokens
            }
        }

        if stream_callback:
            # Gemini streaming
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                params={"key": api_key, "alt": "sse"},
                json=payload,
                stream=True
            )

            # Check for rate limiting errors
            if response.status_code == 429:
                raise ValueError("Google Gemini: Rate limit exceeded (too many requests). Please try again later or use a different provider.")
            elif response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', response.text)
                except:
                    error_msg = response.text
                raise ValueError(f"Google Gemini error: {error_msg}")

            result = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        import json
                        data = json.loads(line_text[6:])
                        if 'candidates' in data:
                            for candidate in data['candidates']:
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    for part in candidate['content']['parts']:
                                        if 'text' in part:
                                            text = part['text']
                                            result += text
                                            stream_callback(text)
            return result
        else:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                params={"key": api_key},
                json=payload
            )

            # Check for rate limiting errors
            if response.status_code == 429:
                raise ValueError("Google Gemini: Rate limit exceeded (too many requests). Please try again later or use a different provider.")

            response.raise_for_status()
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']

    @staticmethod
    def _format_with_xai(raw_text: str, model: str, stream_callback=None) -> str:
        """Format text using xAI (Grok) - OpenAI-compatible API."""
        api_key = current_app.config.get('XAI_API_KEY')
        if not api_key:
            raise ValueError("XAI_API_KEY not configured")

        client = FormatService.get_xai_client()
        max_tokens = FormatService.get_max_output_tokens('xai', model)

        if stream_callback:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FormatService.SYSTEM_PROMPT},
                    {"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                stream=True
            )
            result = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    result += text
                    stream_callback(text)
            return result
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FormatService.SYSTEM_PROMPT},
                    {"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

    @staticmethod
    def _format_with_zhipu(raw_text: str, model: str, stream_callback=None) -> str:
        """Format text using Zhipu AI (智谱AI/BigModel) - OpenAI-compatible API."""
        api_key = current_app.config.get('ZHIPU_API_KEY')
        if not api_key:
            raise ValueError("ZHIPU_API_KEY not configured")

        client = FormatService.get_zhipu_client()
        max_tokens = FormatService.get_max_output_tokens('zhipu', model)

        if stream_callback:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FormatService.SYSTEM_PROMPT},
                    {"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                stream=True
            )
            result = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    result += text
                    stream_callback(text)
            return result
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FormatService.SYSTEM_PROMPT},
                    {"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

    @staticmethod
    def _format_with_lmstudio(raw_text: str, model: str, stream_callback=None, context_length=None) -> str:
        """Format text using LM Studio (OpenAI-compatible API)."""
        client = FormatService.get_lmstudio_client()

        # Build extra params for context length if specified
        extra_body = {}
        if context_length:
            extra_body['num_ctx'] = context_length

        if stream_callback:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FormatService.SYSTEM_PROMPT},
                    {"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}
                ],
                temperature=0.3,
                stream=True,
                extra_body=extra_body if extra_body else None
            )
            result = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    result += text
                    stream_callback(text)
            return result
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FormatService.SYSTEM_PROMPT},
                    {"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}
                ],
                temperature=0.3,
                extra_body=extra_body if extra_body else None
            )
            return response.choices[0].message.content

    @staticmethod
    def get_lmstudio_models():
        """Get available models from LM Studio."""
        try:
            client = FormatService.get_lmstudio_client()
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return []

    @staticmethod
    def infer_context_length_from_model_name(model_id: str) -> int:
        """Infer context length from model name based on known patterns.

        Note: These are conservative defaults for local/quantized models.
        The actual context window may differ - prefer API-reported values when available.
        """
        model_lower = model_id.lower()

        # Qwen models - use conservative defaults for local models
        # While base Qwen 2.5 can support up to 1M, local/quantized versions often have smaller context
        if 'qwen' in model_lower:
            # Check for explicit context size in model name (e.g., qwen2.5-7b-instruct-128k)
            if '1m' in model_lower or '1000k' in model_lower:
                return 1048576
            if '128k' in model_lower:
                return 131072
            if '64k' in model_lower:
                return 65536
            # Default to 32K for local Qwen models - this is safe for most quantized versions
            return 32768

        # Llama models
        if 'llama' in model_lower:
            if '3.2' in model_lower or 'llama-3.2' in model_lower:
                return 131072  # 128K context
            if '3.1' in model_lower or 'llama-3.1' in model_lower:
                return 131072  # 128K context
            if '3' in model_lower:
                return 8192  # 8K for Llama 3
            return 4096  # 4K for older Llama

        # Mistral models
        if 'mistral' in model_lower:
            if 'nemo' in model_lower:
                return 131072  # 128K
            if 'large' in model_lower:
                return 131072  # 128K
            return 32768  # 32K for base Mistral

        # Gemma models
        if 'gemma' in model_lower:
            if '2' in model_lower:
                return 8192
            return 8192

        # Phi models
        if 'phi' in model_lower:
            if '3' in model_lower or 'phi-3' in model_lower:
                return 131072  # 128K
            return 4096

        # DeepSeek models
        if 'deepseek' in model_lower:
            return 131072  # 128K

        # Yi models
        if 'yi' in model_lower:
            if '200k' in model_lower:
                return 200000
            return 32768

        # Default
        return 4096

    @staticmethod
    def get_lmstudio_models_with_info():
        """Get available models from LM Studio with additional info."""
        try:
            base_url = current_app.config.get('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
            # Use the models endpoint to get loaded models
            response = requests.get(f"{base_url}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models_info = []
                for model in data.get('data', []):
                    model_id = model.get('id')
                    # Prefer API-provided context_length (use it if > 0)
                    api_context = model.get('context_length')
                    if api_context and api_context > 0:
                        context_length = api_context
                    else:
                        # Fall back to inference from model name only if API doesn't provide it
                        context_length = FormatService.infer_context_length_from_model_name(model_id)

                    model_info = {
                        'id': model_id,
                        'context_length': context_length,
                        'loaded': True,
                        'context_source': 'api' if (api_context and api_context > 0) else 'inferred'
                    }
                    models_info.append(model_info)
                return models_info
            return []
        except Exception:
            return []

    @staticmethod
    def unload_lmstudio_models():
        """Unload all loaded models from LM Studio."""
        try:
            base_url = current_app.config.get('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
            # LM Studio uses POST /v1/models/unload to unload models
            # First get all loaded models
            response = requests.get(f"{base_url}/models")
            if response.status_code == 200:
                data = response.json()
                for model in data.get('data', []):
                    model_id = model.get('id')
                    if model_id:
                        # Try to unload using the lms endpoint
                        try:
                            requests.post(
                                f"{base_url.replace('/v1', '')}/api/v0/models/unload",
                                json={"model": model_id},
                                timeout=10
                            )
                        except:
                            pass
            return True
        except Exception as e:
            print(f"Error unloading models: {e}")
            return False

    @staticmethod
    def load_lmstudio_model(model_id: str, context_length: int = None):
        """Load a specific model in LM Studio."""
        try:
            base_url = current_app.config.get('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
            payload = {"model": model_id}
            if context_length:
                payload["context_length"] = context_length

            # Try the lms API endpoint for loading
            response = requests.post(
                f"{base_url.replace('/v1', '')}/api/v0/models/load",
                json=payload,
                timeout=120  # Loading can take time
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    @staticmethod
    def get_lmstudio_available_models():
        """Get list of all available (downloadable) models from LM Studio."""
        try:
            base_url = current_app.config.get('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
            response = requests.get(f"{base_url.replace('/v1', '')}/api/v0/models")
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            return []
        except Exception:
            return []

