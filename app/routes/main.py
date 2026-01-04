from flask import Blueprint, render_template
from app.models.transcription import Transcription

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Home page with upload form and recent transcriptions."""
    # Get recent transcriptions
    recent = Transcription.query.order_by(Transcription.created_at.desc()).limit(10).all()
    return render_template('index.html', recent_transcriptions=recent)


@main_bp.route('/transcription/<int:id>')
def view_transcription(id):
    """View a single transcription."""
    transcription = Transcription.query.get_or_404(id)
    return render_template('transcription.html', transcription=transcription)
