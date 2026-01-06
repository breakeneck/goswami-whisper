from flask import Blueprint, render_template, request, jsonify
from app.models.upload import Content
from app.services.embedding_service import EmbeddingService

search_bp = Blueprint('search', __name__)


@search_bp.route('/', methods=['GET'])
def search():
    """Search transcriptions using vector similarity."""
    query = request.args.get('q', '').strip()
    results = []

    if query:
        try:
            # Search in vector database
            vector_results = EmbeddingService.search(query, n_results=20)

            # Get full content data for each result
            seen_ids = set()
            for result in vector_results:
                content_id = result['metadata'].get('content_id')
                if content_id and content_id not in seen_ids:
                    content = Content.query.get(content_id)
                    if content:
                        transcribe = content.transcribe
                        upload = transcribe.upload if transcribe else None
                        results.append({
                            'content': content,
                            'upload': upload,
                            'score': result['score'],
                            'snippet': result['document'][:300] + '...' if len(result['document']) > 300 else result['document']
                        })
                        seen_ids.add(content_id)

        except Exception as e:
            # Fallback to simple text search if vector search fails
            print(f"Vector search failed: {e}, falling back to text search")
            search_term = f"%{query}%"
            contents = Content.query.filter(
                Content.text.like(search_term)
            ).order_by(Content.created_at.desc()).limit(20).all()

            for c in contents:
                transcribe = c.transcribe
                upload = transcribe.upload if transcribe else None
                results.append({
                    'content': c,
                    'upload': upload,
                    'score': None,
                    'snippet': (c.text or '')[:300] + '...'
                })

    # Check if it's an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify([{
            'id': r['content'].id,
            'filename': r['upload'].filename if r['upload'] else None,
            'source_url': r['upload'].source_url if r['upload'] else None,
            'score': r['score'],
            'snippet': r['snippet'],
            'created_at': r['content'].created_at.isoformat()
        } for r in results])

    return render_template('search.html', query=query, results=results)
