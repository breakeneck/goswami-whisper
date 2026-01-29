from openai import OpenAI
from anthropic import Anthropic
from flask import current_app
from app.services.format_service import FormatService


class OpenAIService:
    """Service for formatting text using LLM models (Claude or GPT)."""

    @staticmethod
    def get_openai_client() -> OpenAI:
        """Get OpenAI client with API key from config."""
        return OpenAI(api_key=current_app.config.get('OPENAI_API_KEY'))

    @staticmethod
    def get_anthropic_client() -> Anthropic:
        """Get Anthropic client with API key from config."""
        return Anthropic(api_key=current_app.config.get('ANTHROPIC_API_KEY'))

    @staticmethod
    def format_text(raw_text: str) -> str:
        """
        Format raw transcription text using Claude or GPT.

        Adds proper punctuation, paragraphs, and formatting for readability.
        Uses Claude by default due to larger context window (200K tokens).

        Args:
            raw_text: Raw transcription text

        Returns:
            Formatted text
        """
        if not raw_text:
            return ""

        provider = current_app.config.get('LLM_PROVIDER', 'anthropic')

        prompt = """Please format the following transcription for readability:
1. Add proper punctuation and capitalization
2. Break into logical paragraphs
3. Preserve all the original content and meaning
4. If there are Sanskrit/Hindi terms, keep them as-is
5. Don't add any commentary or summaries

Transcription:
"""

        if provider == 'anthropic':
            return OpenAIService._format_with_claude(raw_text, prompt)
        else:
            return OpenAIService._format_with_openai(raw_text, prompt)

    @staticmethod
    def _format_with_claude(raw_text: str, prompt: str) -> str:
        """Format text using Claude (Anthropic) - 200K context window."""
        client = OpenAIService.get_anthropic_client()
        model = "claude-sonnet-4-20250514"
        max_tokens = FormatService.get_max_output_tokens('anthropic', model)

        response = client.messages.create(
            model=model,
            # max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt + raw_text
                }
            ],
            system="You are a helpful assistant that formats transcriptions for readability while preserving the original content. Output only the formatted transcription, nothing else."
        )

        return response.content[0].text

    @staticmethod
    def _format_with_openai(raw_text: str, prompt: str) -> str:
        """Format text using OpenAI GPT."""
        client = OpenAIService.get_openai_client()
        model = "gpt-4o"
        max_tokens = FormatService.get_max_output_tokens('openai', model)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that formats transcriptions for readability while preserving the original content."
                },
                {
                    "role": "user",
                    "content": prompt + raw_text
                }
            ],
            temperature=0.3,
            # max_completion_tokens=max_tokens
        )

        return response.choices[0].message.content

