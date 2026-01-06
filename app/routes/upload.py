"""Legacy upload routes - deprecated in favor of /api endpoints and wizard workflow."""

from flask import Blueprint, render_template, redirect, url_for, flash

upload_bp = Blueprint('upload', __name__)


@upload_bp.route('/', methods=['GET', 'POST'])
def upload():
    """Legacy upload page - redirects to new wizard interface."""
    flash('The upload page has moved. Please use the new wizard interface.', 'info')
    return redirect(url_for('main.index'))


@upload_bp.route('/processing/<int:id>')
def processing(id):
    """Legacy processing page - redirects to new wizard interface."""
    flash('The processing page has moved. Please use the new wizard interface.', 'info')
    return redirect(url_for('main.index'))


@upload_bp.route('/status/<int:id>')
def status(id):
    """Legacy status endpoint - deprecated."""
    return {'error': 'This endpoint is deprecated. Use /api/transcribes/<id> instead.'}, 410
