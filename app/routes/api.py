"""API routes for managing uploads, transcriptions, and content."""

from flask import Blueprint, request, jsonify, current_app, Response
from werkzeug.utils import secure_filename
import os
import uuid
import threading
import json
import time
from app import db
from app.models.upload import Upload, Transcribe, Content
from app.services.transcribe_service import TranscribeService
from app.services.format_service import FormatService
from app.services.embedding_service import EmbeddingService

api_bp = Blueprint('api', __name__)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


# ============ Upload Endpoints ============

@api_bp.route('/uploads', methods=['GET'])
def list_uploads():
    """List all uploads with their nested data."""
    uploads = Upload.query.order_by(Upload.created_at.desc()).all()
    return jsonify([u.to_dict() for u in uploads])


@api_bp.route('/uploads', methods=['POST'])
def create_upload():
    """Create a new upload from file or URL."""
    try:
        file_path = None
        original_filename = None
        source_url = None

        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']

            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400

            original_filename = file.filename
            filename = secure_filename(file.filename)

            # Generate unique filename if needed
            if not filename or filename.startswith('.'):
                original_ext = ''
                if '.' in file.filename:
                    original_ext = '.' + file.filename.rsplit('.', 1)[1].lower()
                filename = f"{uuid.uuid4().hex}{original_ext}"
            else:
                # Add UUID to prevent conflicts
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"

            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_size = os.path.getsize(file_path)

        elif request.form.get('url'):
            source_url = request.form.get('url').strip()

            # Download the file
            file_path = TranscribeService.download_from_url(
                source_url,
                current_app.config['UPLOAD_FOLDER']
            )
            original_filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        else:
            return jsonify({'error': 'Please provide a file or URL'}), 400

        # Get audio duration
        duration = TranscribeService.get_audio_duration(file_path) if file_path else None

        # Create upload record
        upload = Upload(
            filename=os.path.basename(file_path) if file_path else original_filename,
            original_filename=original_filename,
            file_path=file_path,
            source_url=source_url,
            file_size=file_size if 'file_size' in dir() else None,
            duration_seconds=duration
        )
        db.session.add(upload)
        db.session.commit()

        return jsonify(upload.to_dict()), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/uploads/<int:upload_id>', methods=['DELETE'])
def delete_upload(upload_id):
    """Delete an upload and all related data."""
    upload = Upload.query.get_or_404(upload_id)

    # Remove from ChromaDB if indexed
    if upload.is_indexed:
        try:
            EmbeddingService.delete_by_upload_id(upload_id)
        except Exception as e:
            print(f"Error removing from ChromaDB: {e}")

    # Delete file from disk
    if upload.file_path and os.path.exists(upload.file_path):
        try:
            os.remove(upload.file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")

    db.session.delete(upload)
    db.session.commit()

    return jsonify({'message': 'Upload deleted'}), 200


# ============ Transcribe Endpoints ============

@api_bp.route('/transcribe', methods=['POST'])
def start_transcription():
    """Start a new transcription for an upload."""
    data = request.get_json()
    upload_id = data.get('upload_id')
    provider = data.get('provider', 'whisper')
    model = data.get('model', 'base')

    upload = Upload.query.get_or_404(upload_id)

    # Create transcribe record
    transcribe = Transcribe(
        upload_id=upload_id,
        provider=provider,
        model=model,
        status='pending'
    )
    db.session.add(transcribe)
    db.session.commit()

    # Start async processing
    app = current_app._get_current_object()
    thread = threading.Thread(
        target=process_transcription_async,
        args=(app, transcribe.id, upload.file_path, provider, model)
    )
    thread.daemon = True
    thread.start()

    return jsonify(transcribe.to_dict()), 201


def process_transcription_async(app, transcribe_id, file_path, provider, model):
    """Process transcription in background thread."""
    with app.app_context():
        transcribe = Transcribe.query.get(transcribe_id)
        if not transcribe:
            return

        try:
            transcribe.status = 'processing'
            transcribe.progress = 0.0
            db.session.commit()

            start_time = time.time()

            def progress_callback(progress):
                # Update progress in database
                from sqlalchemy import text
                with db.engine.connect() as conn:
                    conn.execute(
                        text("UPDATE transcribes SET progress = :progress WHERE id = :id"),
                        {"progress": progress, "id": transcribe_id}
                    )
                    conn.commit()

            text_result = TranscribeService.transcribe(
                file_path,
                provider,
                model,
                progress_callback=progress_callback
            )

            duration = time.time() - start_time

            transcribe.text = text_result
            transcribe.status = 'completed'
            transcribe.progress = 100.0
            transcribe.duration_seconds = duration
            db.session.commit()

        except Exception as e:
            transcribe.status = 'error'
            transcribe.error_message = str(e)
            db.session.commit()


@api_bp.route('/transcribe/<int:transcribe_id>/status', methods=['GET'])
def get_transcription_status(transcribe_id):
    """Get the status of a transcription."""
    db.session.expire_all()
    transcribe = Transcribe.query.get_or_404(transcribe_id)
    return jsonify(transcribe.to_dict())


@api_bp.route('/transcribe/<int:transcribe_id>', methods=['DELETE'])
def delete_transcription(transcribe_id):
    """Delete a transcription and all related content."""
    transcribe = Transcribe.query.get_or_404(transcribe_id)

    # Check if any content is indexed
    for content in transcribe.contents:
        if content.is_indexed:
            try:
                EmbeddingService.delete_by_content_id(content.id)
            except Exception as e:
                print(f"Error removing content from ChromaDB: {e}")

    db.session.delete(transcribe)
    db.session.commit()

    return jsonify({'message': 'Transcription deleted'}), 200


# ============ Content (Format) Endpoints ============

@api_bp.route('/format', methods=['POST'])
def start_formatting():
    """Start formatting a transcription."""
    data = request.get_json()
    transcribe_id = data.get('transcribe_id')
    provider = data.get('provider', 'anthropic')
    model = data.get('model', 'claude-sonnet-4-20250514')

    transcribe = Transcribe.query.get_or_404(transcribe_id)

    if not transcribe.text:
        return jsonify({'error': 'Transcription has no text to format'}), 400

    # Create content record
    content = Content(
        transcribe_id=transcribe_id,
        provider=provider,
        model=model,
        status='pending'
    )
    db.session.add(content)
    db.session.commit()

    # Start async processing
    app = current_app._get_current_object()
    thread = threading.Thread(
        target=process_formatting_async,
        args=(app, content.id, transcribe.text, provider, model)
    )
    thread.daemon = True
    thread.start()

    return jsonify(content.to_dict()), 201


def process_formatting_async(app, content_id, raw_text, provider, model):
    """Process formatting in background thread."""
    with app.app_context():
        content = Content.query.get(content_id)
        if not content:
            return

        try:
            content.status = 'processing'
            db.session.commit()

            start_time = time.time()
            formatted_text = FormatService.format_text(raw_text, provider, model)
            duration = time.time() - start_time

            content.text = formatted_text
            content.status = 'completed'
            content.duration_seconds = duration
            db.session.commit()

        except Exception as e:
            content.status = 'error'
            content.error_message = str(e)
            db.session.commit()


@api_bp.route('/format/<int:content_id>/stream', methods=['POST'])
def format_with_streaming(content_id):
    """Format content with streaming response."""
    content = Content.query.get_or_404(content_id)
    transcribe = Transcribe.query.get_or_404(content.transcribe_id)

    if not transcribe.text:
        return jsonify({'error': 'Transcription has no text to format'}), 400

    content.status = 'processing'
    db.session.commit()

    def generate():
        try:
            full_text = []
            start_time = time.time()

            def stream_callback(text_chunk):
                full_text.append(text_chunk)
                yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

            FormatService.format_text(
                transcribe.text,
                content.provider,
                content.model,
                stream_callback=stream_callback
            )

            duration = time.time() - start_time

            # Save the final text
            final_text = ''.join(full_text)
            with current_app.app_context():
                content_db = Content.query.get(content_id)
                content_db.text = final_text
                content_db.status = 'completed'
                content_db.duration_seconds = duration
                db.session.commit()

            yield f"data: {json.dumps({'done': True, 'duration_seconds': duration})}\n\n"

        except Exception as e:
            with current_app.app_context():
                content_db = Content.query.get(content_id)
                content_db.status = 'error'
                content_db.error_message = str(e)
                db.session.commit()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@api_bp.route('/format/<int:content_id>/status', methods=['GET'])
def get_format_status(content_id):
    """Get the status of a formatting job."""
    db.session.expire_all()
    content = Content.query.get_or_404(content_id)
    return jsonify(content.to_dict())


@api_bp.route('/format/<int:content_id>', methods=['DELETE'])
def delete_content(content_id):
    """Delete formatted content."""
    content = Content.query.get_or_404(content_id)

    if content.is_indexed:
        try:
            EmbeddingService.delete_by_content_id(content_id)
        except Exception as e:
            print(f"Error removing from ChromaDB: {e}")

    db.session.delete(content)
    db.session.commit()

    return jsonify({'message': 'Content deleted'}), 200


# ============ Indexing Endpoints ============

@api_bp.route('/index', methods=['POST'])
def start_indexing():
    """Index formatted content to ChromaDB."""
    data = request.get_json()
    content_id = data.get('content_id')

    content = Content.query.get_or_404(content_id)
    transcribe = Transcribe.query.get_or_404(content.transcribe_id)
    upload = Upload.query.get_or_404(transcribe.upload_id)

    if not content.text:
        return jsonify({'error': 'Content has no text to index'}), 400

    if upload.is_indexed:
        return jsonify({'error': 'This upload is already indexed'}), 400

    try:
        start_time = time.time()

        # Index the content
        EmbeddingService.index_content(
            upload_id=upload.id,
            content_id=content.id,
            text=content.text,
            metadata={
                'filename': upload.filename,
                'transcribe_provider': transcribe.provider,
                'transcribe_model': transcribe.model,
                'format_provider': content.provider,
                'format_model': content.model
            }
        )

        duration = time.time() - start_time

        # Mark as indexed
        content.is_indexed = True
        upload.is_indexed = True
        upload.index_duration_seconds = duration
        db.session.commit()

        return jsonify({'message': 'Content indexed successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/index/<int:upload_id>', methods=['DELETE'])
def remove_index(upload_id):
    """Remove indexed content for an upload."""
    upload = Upload.query.get_or_404(upload_id)

    if not upload.is_indexed:
        return jsonify({'error': 'Upload is not indexed'}), 400

    try:
        EmbeddingService.delete_by_upload_id(upload_id)

        # Update indexed flags
        upload.is_indexed = False
        for transcribe in upload.transcribes:
            for content in transcribe.contents:
                content.is_indexed = False
        db.session.commit()

        return jsonify({'message': 'Index removed successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ Configuration Endpoints ============

@api_bp.route('/config/providers', methods=['GET'])
def get_providers():
    """Get available providers and models."""
    config = {
        'transcribe': current_app.config.get('TRANSCRIBE_PROVIDERS', {}),
        'format': current_app.config.get('FORMAT_PROVIDERS', {})
    }

    # Try to get LM Studio models dynamically
    try:
        lmstudio_models = FormatService.get_lmstudio_models()
        if lmstudio_models:
            config['format']['lmstudio']['models'] = lmstudio_models
    except Exception:
        pass

    return jsonify(config)


# ============ Search Endpoints ============

@api_bp.route('/search', methods=['GET'])
def search():
    """Search indexed content."""
    query = request.args.get('q', '').strip()
    n_results = request.args.get('limit', 20, type=int)

    if not query:
        return jsonify([])

    try:
        results = EmbeddingService.search(query, n_results=n_results)

        # Enrich results with upload info
        enriched_results = []
        seen_uploads = set()

        for result in results:
            upload_id = result['metadata'].get('upload_id')
            if upload_id and upload_id not in seen_uploads:
                upload = Upload.query.get(upload_id)
                if upload:
                    enriched_results.append({
                        'upload': upload.to_dict(),
                        'score': result['score'],
                        'snippet': result['document'][:300] + '...' if len(result['document']) > 300 else result['document']
                    })
                    seen_uploads.add(upload_id)

        return jsonify(enriched_results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

