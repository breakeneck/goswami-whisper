from flask import Blueprint, render_template, request, jsonify
from app.models.transcription import Transcription
from app.services.embedding_service import EmbeddingService

search_bp = Blueprint('search', __name__)


@search_bp.route('/', methods=['GET'])
def search():
    """Search transcriptions using vector similarity."""
    query = request.args.get('q', '').strip()
    results = []
    transcriptions = []

    if query:
        try:
            # Search in vector database
            vector_results = EmbeddingService.search(query, n_results=20)

            # Get full transcription data for each result
            seen_ids = set()
            for result in vector_results:
                transcription_id = result['metadata'].get('transcription_id')
                if transcription_id and transcription_id not in seen_ids:
                    transcription = Transcription.query.get(transcription_id)
                    if transcription:
                        transcriptions.append({
                            'transcription': transcription,
                            'score': result['score'],
                            'snippet': result['document'][:300] + '...' if len(result['document']) > 300 else result['document']
                        })
                        seen_ids.add(transcription_id)

            results = transcriptions

        except Exception as e:
            # Fallback to simple text search if vector search fails
            print(f"Vector search failed: {e}, falling back to text search")
            search_term = f"%{query}%"
            transcriptions_db = Transcription.query.filter(
                (Transcription.raw_text.like(search_term)) |
                (Transcription.formatted_text.like(search_term)) |
                (Transcription.filename.like(search_term))
            ).order_by(Transcription.created_at.desc()).limit(20).all()

            results = [{
                'transcription': t,
                'score': None,
                'snippet': (t.formatted_text or t.raw_text or '')[:300] + '...'
            } for t in transcriptions_db]

    # Check if it's an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify([{
            'id': r['transcription'].id,
            'filename': r['transcription'].filename,
            'source_url': r['transcription'].source_url,
            'score': r['score'],
            'snippet': r['snippet'],
            'created_at': r['transcription'].created_at.isoformat()
        } for r in results])

    return render_template('search.html', query=query, results=results)

