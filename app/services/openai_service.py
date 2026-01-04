from openai import OpenAI
from flask import current_app


class OpenAIService:
    """Service for formatting text using OpenAI GPT models."""

    @staticmethod
    def get_client() -> OpenAI:
        """Get OpenAI client with API key from config."""
        return OpenAI(api_key=current_app.config.get('OPENAI_API_KEY'))

    @staticmethod
    def format_text(raw_text: str) -> str:
        """
        Format raw transcription text using OpenAI GPT-4.

        Adds proper punctuation, paragraphs, and formatting for readability.

        Args:
            raw_text: Raw transcription text

        Returns:
            Formatted text
        """
        if not raw_text:
            return ""

        client = OpenAIService.get_client()

        prompt = """Please format the following transcription for readability:
1. Add proper punctuation and capitalization
2. Break into logical paragraphs
3. Preserve all the original content and meaning
4. If there are Sanskrit/Hindi terms, keep them as-is
5. Don't add any commentary or summaries

Transcription:
"""

        response = client.chat.completions.create(
            model="gpt-4o",
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
            max_tokens=4096
        )

        return response.choices[0].message.content

