from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from werkzeug.utils import secure_filename
import os
import threading
import uuid
from app import db
from app.models.transcription import Transcription
from app.services.whisper_service import WhisperService
from app.services.openai_service import OpenAIService
from app.services.embedding_service import EmbeddingService

upload_bp = Blueprint('upload', __name__)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def process_transcription_async(app, transcription_id, file_path, model_name):
    """Process transcription in background thread."""
    with app.app_context():
        transcription = Transcription.query.get(transcription_id)
        if not transcription:
            return

        try:
            # Transcribe with Whisper
            transcription.status = 'transcribing'
            transcription.progress = 0.0
            db.session.commit()

            raw_text = WhisperService.transcribe_file(
                file_path,
                model_name=model_name,
                transcription_id=transcription_id,
                app=app
            )
            transcription.raw_text = raw_text
            transcription.status = 'formatting'
            transcription.progress = 100.0
            db.session.commit()
        except Exception as e:
            transcription.status = 'error'
            transcription.error_message = f"Transcription error: {str(e)}"
            db.session.commit()
            return

        # Format with OpenAI
        try:
            formatted_text = OpenAIService.format_text(raw_text)
            transcription.formatted_text = formatted_text
            transcription.status = 'indexing'
            db.session.commit()
        except Exception as e:
            transcription.status = 'completed'
            transcription.error_message = f"Formatting skipped: {str(e)}"
            db.session.commit()

        # Add to vector database
        try:
            text_to_embed = transcription.formatted_text or transcription.raw_text
            if text_to_embed:
                # Update progress to show indexing is in progress
                transcription.progress = 50.0
                db.session.commit()

                EmbeddingService.index_transcription(
                    transcription_id=transcription.id,
                    text=text_to_embed
                )

                # Update progress to show indexing is complete
                transcription.progress = 100.0
                db.session.commit()
            transcription.status = 'completed'
            db.session.commit()
        except Exception as e:
            transcription.status = 'completed'
            transcription.error_message = f"Indexing warning: {str(e)}"
            db.session.commit()


@upload_bp.route('/', methods=['GET', 'POST'])
def upload():
    """Handle file upload or URL submission."""
    if request.method == 'POST':
        transcription = None
        file_path = None
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.best == 'application/json'

        try:
            # Get whisper model from form
            model_name = request.form.get('whisper_model', current_app.config['WHISPER_MODEL'])

            # Check if it's a file upload or URL
            if 'file' in request.files and request.files['file'].filename:
                file = request.files['file']

                if not allowed_file(file.filename):
                    if is_ajax:
                        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(current_app.config['ALLOWED_EXTENSIONS'])}), 400
                    flash('Invalid file type. Allowed types: ' + ', '.join(current_app.config['ALLOWED_EXTENSIONS']), 'error')
                    return redirect(url_for('upload.upload'))

                # Save the file
                original_filename = file.filename
                filename = secure_filename(file.filename)
                # If secure_filename removed all characters (non-ASCII filename),
                # generate a UUID-based filename preserving the extension
                if not filename or filename.startswith('.'):
                    original_ext = ''
                    if '.' in file.filename:
                        original_ext = '.' + file.filename.rsplit('.', 1)[1].lower()
                    filename = f"{uuid.uuid4().hex}{original_ext}"
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Create transcription record
                transcription = Transcription(
                    filename=original_filename,  # Store original filename for display
                    status='pending',
                    progress=0.0
                )
                db.session.add(transcription)
                db.session.commit()

            elif request.form.get('url'):
                url = request.form.get('url').strip()

                # Create transcription record
                transcription = Transcription(
                    source_url=url,
                    status='downloading',
                    progress=0.0
                )
                db.session.add(transcription)
                db.session.commit()

                # Download the file
                try:
                    file_path = WhisperService.download_from_url(
                        url,
                        current_app.config['UPLOAD_FOLDER']
                    )
                    transcription.filename = os.path.basename(file_path)
                    transcription.status = 'pending'
                    db.session.commit()
                except Exception as e:
                    transcription.status = 'error'
                    transcription.error_message = str(e)
                    db.session.commit()
                    if is_ajax:
                        return jsonify({'error': f'Error downloading from URL: {str(e)}'}), 400
                    flash(f'Error downloading from URL: {str(e)}', 'error')
                    return redirect(url_for('upload.upload'))
            else:
                if is_ajax:
                    return jsonify({'error': 'Please provide a file or URL'}), 400
                flash('Please provide a file or URL', 'error')
                return redirect(url_for('upload.upload'))

            # Start async processing in background thread
            app = current_app._get_current_object()
            thread = threading.Thread(
                target=process_transcription_async,
                args=(app, transcription.id, file_path, model_name)
            )
            thread.daemon = True
            thread.start()

            # Return JSON response for AJAX or redirect
            if is_ajax:
                return jsonify({
                    'id': transcription.id,
                    'status': 'processing',
                    'message': 'Transcription started'
                })

            # For non-AJAX, redirect to a processing page
            return redirect(url_for('upload.processing', id=transcription.id))

        except Exception as e:
            if transcription:
                transcription.status = 'error'
                transcription.error_message = str(e)
                db.session.commit()
            if is_ajax:
                return jsonify({'error': str(e)}), 500
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('upload.upload'))

    return render_template('upload.html',
                           whisper_models=current_app.config['WHISPER_MODELS'],
                           default_model=current_app.config['WHISPER_MODEL'])


@upload_bp.route('/processing/<int:id>')
def processing(id):
    """Show processing page with progress indicator."""
    transcription = Transcription.query.get_or_404(id)
    return render_template('processing.html', transcription=transcription)


@upload_bp.route('/status/<int:id>')
def status(id):
    """Get the status of a transcription (for AJAX polling)."""
    # Expire all objects to force a fresh read from the database
    db.session.expire_all()
    transcription = Transcription.query.get_or_404(id)
    return jsonify(transcription.to_dict())

