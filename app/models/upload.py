from datetime import datetime
from app import db


class Upload(db.Model):
    """Model for storing uploaded files."""

    __tablename__ = 'uploads'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(500), nullable=False)
    original_filename = db.Column(db.String(500), nullable=True)
    file_path = db.Column(db.String(1000), nullable=True)
    source_url = db.Column(db.Text, nullable=True)
    file_size = db.Column(db.BigInteger, nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)
    is_indexed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    transcribes = db.relationship('Transcribe', backref='upload', lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Upload {self.id}: {self.filename}>'

    def to_dict(self):
        """Convert upload to dictionary."""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'source_url': self.source_url,
            'file_size': self.file_size,
            'duration_seconds': self.duration_seconds,
            'is_indexed': self.is_indexed,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'transcribes': [t.to_dict() for t in self.transcribes.all()]
        }


class Transcribe(db.Model):
    """Model for storing transcribed text with provider info."""

    __tablename__ = 'transcribes'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('uploads.id'), nullable=False)
    text = db.Column(db.Text, nullable=True)
    provider = db.Column(db.String(50), nullable=False)  # whisper, faster-whisper
    model = db.Column(db.String(50), nullable=False)  # tiny, base, small, medium, large
    status = db.Column(db.String(50), default='pending')  # pending, processing, completed, error
    progress = db.Column(db.Float, default=0.0)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    contents = db.relationship('Content', backref='transcribe', lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Transcribe {self.id}: {self.provider}/{self.model}>'

    def to_dict(self):
        """Convert transcribe to dictionary."""
        return {
            'id': self.id,
            'upload_id': self.upload_id,
            'text': self.text,
            'provider': self.provider,
            'model': self.model,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'contents': [c.to_dict() for c in self.contents.all()]
        }


class Content(db.Model):
    """Model for storing formatted content with provider info."""

    __tablename__ = 'contents'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    transcribe_id = db.Column(db.Integer, db.ForeignKey('transcribes.id'), nullable=False)
    text = db.Column(db.Text, nullable=True)
    provider = db.Column(db.String(50), nullable=False)  # openai, anthropic, gemini, lmstudio
    model = db.Column(db.String(100), nullable=False)  # gpt-4o, claude-sonnet-4-20250514, etc.
    status = db.Column(db.String(50), default='pending')  # pending, processing, completed, error
    progress = db.Column(db.Float, default=0.0)
    error_message = db.Column(db.Text, nullable=True)
    is_indexed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Content {self.id}: {self.provider}/{self.model}>'

    def to_dict(self):
        """Convert content to dictionary."""
        return {
            'id': self.id,
            'transcribe_id': self.transcribe_id,
            'text': self.text,
            'provider': self.provider,
            'model': self.model,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'is_indexed': self.is_indexed,
            'duration_seconds': self.duration_seconds,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

