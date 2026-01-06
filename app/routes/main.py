from flask import Blueprint, render_template, abort
from app.models.upload import Transcribe, Content

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Home page with uploads table and wizard controls."""
    return render_template('index_new.html')


@main_bp.route('/transcription/<int:id>')
def view_transcription(id):
    """View a single transcription - redirects to content view."""
    # Legacy route - try to find corresponding content
    content = Content.query.get(id)
    if content:
        return render_template('transcription.html', transcription=content)
    # Try to find transcribe
    transcribe = Transcribe.query.get(id)
    if transcribe:
        return render_template('transcription.html', transcription=transcribe)
    abort(404)

