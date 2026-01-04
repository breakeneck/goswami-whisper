# Goswami Whisper

A Flask web application for transcribing and searching Vedic lectures (Krishna consciousness lectures) with AI-powered tools.

## Features

- **Audio/Video Upload**: Upload files or provide URLs (YouTube, etc.)
- **Whisper Transcription**: Accurate speech-to-text using OpenAI Whisper CLI
- **AI Formatting**: Intelligent formatting with GPT-4, optimized for Vedic/Sanskrit terminology
- **MySQL Storage**: Persistent storage of raw and formatted transcriptions
- **Semantic Search**: LaBSE-powered vector search using ChromaDB for RAG capabilities

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for MySQL)
- Whisper CLI installed (`pip install openai-whisper`)
- NVIDIA GPU (recommended, the app is optimized for RTX 3090)
- FFmpeg (for audio processing)

## Installation

1. **Clone the repository**:
   ```bash
   cd goswami-whisper
   ```

2. **Start MySQL with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and set your OPENAI_API_KEY
   ```

6. **Run the application**:
   ```bash
   python main.py
   ```

7. **Access the web interface**:
   Open http://localhost:5000 in your browser

## Configuration

Edit the `.env` file to configure:

- `OPENAI_API_KEY`: Your OpenAI API key for text formatting
- `WHISPER_MODEL`: Whisper model size (tiny, base, small, medium, large)
- MySQL credentials (default values work with Docker Compose)

## Project Structure

```
goswami-whisper/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── models/
│   │   └── transcription.py  # SQLAlchemy models
│   ├── routes/
│   │   ├── main.py           # Home and view routes
│   │   ├── upload.py         # File/URL upload handling
│   │   └── search.py         # Vector search
│   ├── services/
│   │   ├── whisper_service.py    # Whisper CLI integration
│   │   ├── openai_service.py     # OpenAI API formatting
│   │   └── embedding_service.py  # LaBSE embeddings & ChromaDB
│   └── templates/            # Jinja2 HTML templates
├── uploads/                  # Uploaded files
├── chroma_db/               # ChromaDB vector storage
├── config.py                # Configuration
├── main.py                  # Application entry point
├── docker-compose.yml       # MySQL container
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables
```

## Usage

1. **Upload a recording**:
   - Go to the Upload page
   - Either upload an audio/video file or paste a YouTube URL
   - Wait for processing (transcription → formatting → indexing)

2. **View transcriptions**:
   - Browse recent transcriptions on the home page
   - Click to view full formatted and raw text

3. **Search lectures**:
   - Use the search bar in the navigation
   - Semantic search finds relevant content even with different wording

## Technical Details

- **Whisper**: Uses OpenAI Whisper CLI with configurable model size
- **OpenAI GPT-4**: Formats transcriptions preserving Vedic/Sanskrit terminology
- **LaBSE**: Language-agnostic BERT embeddings for multilingual search
- **ChromaDB**: Local vector database for persistent embedding storage
- **MySQL 8.0**: Relational database for transcription metadata

## License

MIT License

