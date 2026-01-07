"""Service for formatting transcribed text using various LLM providers."""

from openai import OpenAI
from anthropic import Anthropic
from flask import current_app
import requests


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
        elif provider == 'lmstudio':
            return FormatService._format_with_lmstudio(raw_text, model, stream_callback, context_length)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def _format_with_openai(raw_text: str, model: str, stream_callback=None) -> str:
        """Format text using OpenAI GPT."""
        client = FormatService.get_openai_client()

        if stream_callback:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FormatService.SYSTEM_PROMPT},
                    {"role": "user", "content": FormatService.FORMAT_PROMPT + raw_text}
                ],
                temperature=0.3,
                max_tokens=16384,
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
                max_tokens=16384
            )
            return response.choices[0].message.content

    @staticmethod
    def _format_with_anthropic(raw_text: str, model: str, stream_callback=None) -> str:
        """Format text using Claude (Anthropic)."""
        client = FormatService.get_anthropic_client()

        if stream_callback:
            with client.messages.stream(
                model=model,
                max_tokens=16384,
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
                max_tokens=16384,
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

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        payload = {
            "contents": [{
                "parts": [{"text": FormatService.SYSTEM_PROMPT + "\n\n" + FormatService.FORMAT_PROMPT + raw_text}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 16384
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

