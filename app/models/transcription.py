from datetime import datetime
from app import db


class Transcription(db.Model):
    """Model for storing transcription data (legacy)."""

    __tablename__ = 'transcriptions'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=True)
    source_url = db.Column(db.Text, nullable=True)
    raw_text = db.Column(db.Text, nullable=True)
    formatted_text = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(50), default='pending')
    error_message = db.Column(db.Text, nullable=True)
    progress = db.Column(db.Float, default=0.0)  # 0.0 to 100.0
    duration_seconds = db.Column(db.Float, nullable=True)
    current_position = db.Column(db.Float, default=0.0)  # Current position in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Transcription {self.id}: {self.filename or self.source_url}>'

    def to_dict(self):
        """Convert transcription to dictionary."""
        return {
            'id': self.id,
            'filename': self.filename,
            'source_url': self.source_url,
            'raw_text': self.raw_text,
            'formatted_text': self.formatted_text,
            'status': self.status,
            'error_message': self.error_message,
            'progress': self.progress or 0.0,
            'duration_seconds': self.duration_seconds,
            'current_position': self.current_position or 0.0,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

