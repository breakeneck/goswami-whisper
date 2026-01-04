"""
Goswami Whisper - Flask Application for Transcription and Search

A web interface to:
- Upload audio/video files or URLs
- Transcribe using Whisper
- Format with OpenAI GPT-4
- Store in MySQL database
- Search using LaBSE embeddings and ChromaDB

Usage:
    1. Start MySQL with Docker Compose: docker-compose up -d
    2. Install dependencies: pip install -r requirements.txt
    3. Set your OpenAI API key in .env file
    4. Run the app: python main.py
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    print("=" * 60)
    print("Goswami Whisper - Vedic Lecture Transcription System")
    print("=" * 60)
    print("\nMake sure MySQL is running: docker-compose up -d")
    print("Access the web interface at: http://localhost:5000")
    print("\n" + "=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)
