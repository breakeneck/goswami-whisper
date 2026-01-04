from flask import Blueprint, render_template
    return render_template('transcription.html', transcription=transcription)
    transcription = Transcription.query.get_or_404(id)
    """View a single transcription."""
def view_transcription(id):
@main_bp.route('/transcription/<int:id>')


    return render_template('index.html', recent_transcriptions=recent)
    recent = Transcription.query.order_by(Transcription.created_at.desc()).limit(10).all()
    # Get recent transcriptions
    """Home page with upload form and recent transcriptions."""
def index():
@main_bp.route('/')


main_bp = Blueprint('main', __name__)

from app.models.transcription import Transcription

