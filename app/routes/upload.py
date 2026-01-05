from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from werkzeug.utils import secure_filename
import os
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


@upload_bp.route('/', methods=['GET', 'POST'])
def upload():
    """Handle file upload or URL submission."""
    if request.method == 'POST':
        transcription = None
        file_path = None

        try:
            # Check if it's a file upload or URL
            if 'file' in request.files and request.files['file'].filename:
                file = request.files['file']

                if not allowed_file(file.filename):
                    flash('Invalid file type. Allowed types: ' + ', '.join(current_app.config['ALLOWED_EXTENSIONS']), 'error')
                    return redirect(url_for('upload.upload'))

                # Save the file
                filename = secure_filename(file.filename)
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Create transcription record
                transcription = Transcription(
                    filename=filename,
                    status='transcribing'
                )
                db.session.add(transcription)
                db.session.commit()

            elif request.form.get('url'):
                url = request.form.get('url').strip()

                # Create transcription record
                transcription = Transcription(
                    source_url=url,
                    status='downloading'
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
                    transcription.status = 'transcribing'
                    db.session.commit()
                except Exception as e:
                    transcription.status = 'error'
                    transcription.error_message = str(e)
                    db.session.commit()
                    flash(f'Error downloading from URL: {str(e)}', 'error')
                    return redirect(url_for('upload.upload'))
            else:
                flash('Please provide a file or URL', 'error')
                return redirect(url_for('upload.upload'))

            # Transcribe with Whisper
            try:
                raw_text = WhisperService.transcribe_file(file_path)
                transcription.raw_text = raw_text
                transcription.status = 'formatting'
                db.session.commit()
            except Exception as e:
                transcription.status = 'error'
                transcription.error_message = f"Transcription error: {str(e)}"
                db.session.commit()
                flash(f'Error during transcription: {str(e)}', 'error')
                return redirect(url_for('main.view_transcription', id=transcription.id))

            # Format with OpenAI
            try:
                formatted_text = OpenAIService.format_text(raw_text)
                transcription.formatted_text = formatted_text
                transcription.status = 'indexing'
                db.session.commit()
            except Exception as e:
                # Still save even if formatting fails
                transcription.status = 'completed'
                transcription.error_message = f"Formatting skipped: {str(e)}"
                db.session.commit()
                flash(f'Transcription completed but formatting failed: {str(e)}', 'warning')

            # Add to vector database
            try:
                text_to_embed = transcription.formatted_text or transcription.raw_text
                if text_to_embed:
                    EmbeddingService.index_transcription(
                        transcription_id=transcription.id,
                        text=text_to_embed
                    )
                transcription.status = 'completed'
                db.session.commit()
                flash('Transcription completed successfully!', 'success')
            except Exception as e:
                transcription.status = 'completed'
                transcription.error_message = f"Indexing warning: {str(e)}"
                db.session.commit()
                flash(f'Transcription completed but indexing failed: {str(e)}', 'warning')

            return redirect(url_for('main.view_transcription', id=transcription.id))

        except Exception as e:
            if transcription:
                transcription.status = 'error'
                transcription.error_message = str(e)
                db.session.commit()
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('upload.upload'))

    return render_template('upload.html',
                           whisper_models=current_app.config['WHISPER_MODELS'],
                           default_model=current_app.config['WHISPER_MODEL'])


@upload_bp.route('/status/<int:id>')
def status(id):
    """Get the status of a transcription (for AJAX polling)."""
    transcription = Transcription.query.get_or_404(id)
    return jsonify(transcription.to_dict())

