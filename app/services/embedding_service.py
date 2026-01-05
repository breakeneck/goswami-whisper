import chromadb
from sentence_transformers import SentenceTransformer
from flask import current_app
from typing import List, Dict, Any
import os


class EmbeddingService:
    """Service for generating embeddings and searching using ChromaDB."""

    _model = None
    _client = None
    _collection = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Get or initialize the sentence transformer model."""
        if cls._model is None:
            # LaBSE is good for multilingual content
            cls._model = SentenceTransformer('sentence-transformers/LaBSE')
        return cls._model

    @classmethod
    def get_collection(cls):
        """Get or initialize the ChromaDB collection."""
        if cls._collection is None:
            persist_dir = current_app.config.get('CHROMA_PERSIST_DIR', 'chroma_db')
            os.makedirs(persist_dir, exist_ok=True)

            cls._client = chromadb.PersistentClient(path=persist_dir)
            # IMPORTANT: Set embedding_function=None to prevent ChromaDB from
            # loading its own default model (all-MiniLM-L6-v2).
            # We provide pre-computed embeddings from LaBSE, so no embedding function needed.
            cls._collection = cls._client.get_or_create_collection(
                name="transcriptions",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
        return cls._collection

    @classmethod
    def chunk_text(cls, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for embedding.

        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence-ending punctuation
                for punct in ['. ', '? ', '! ', '\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start:
                        end = last_punct + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    @classmethod
    def index_transcription(cls, transcription_id: int, text: str) -> None:
        """
        Index a transcription in ChromaDB.

        Args:
            transcription_id: ID of the transcription record
            text: Text to index
        """
        if not text:
            return

        model = cls.get_model()
        collection = cls.get_collection()

        # Split text into chunks
        chunks = cls.chunk_text(text)

        if not chunks:
            return

        # Generate embeddings
        embeddings = model.encode(chunks).tolist()

        # Prepare documents for ChromaDB
        ids = [f"{transcription_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"transcription_id": transcription_id, "chunk_index": i} for i in range(len(chunks))]

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )


    @classmethod
    def search(cls, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar transcriptions.

        Args:
            query: Search query
            n_results: Maximum number of results

        Returns:
            List of search results with scores and metadata
        """
        model = cls.get_model()
        collection = cls.get_collection()

        # Generate query embedding
        query_embedding = model.encode([query]).tolist()[0]

        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': doc_id,
                    'document': results['documents'][0][i] if results['documents'] else '',
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': 1 - results['distances'][0][i] if results['distances'] else 0
                })

        return formatted_results

    @classmethod
    def delete_transcription(cls, transcription_id: int) -> None:
        """
        Delete all indexed chunks for a transcription.

        Args:
            transcription_id: ID of the transcription to delete
        """
        collection = cls.get_collection()

        # Find all chunks for this transcription
        results = collection.get(
            where={"transcription_id": transcription_id}
        )

        if results['ids']:
            collection.delete(ids=results['ids'])

