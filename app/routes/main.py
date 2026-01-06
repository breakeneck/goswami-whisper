from flask import Blueprint, render_template
from app.models.transcription import Transcription
from app.models.upload import Upload

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Home page with uploads table and wizard controls."""
    return render_template('index_new.html')


@main_bp.route('/transcription/<int:id>')
def view_transcription(id):
    """View a single transcription (legacy support)."""
    transcription = Transcription.query.get_or_404(id)
    return render_template('transcription.html', transcription=transcription)
