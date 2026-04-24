"""Input validation helpers shared between app.py and batch_predict.py."""

from pathlib import Path
from typing import Optional

import config


class ValidationError(ValueError):
    """Raised when an uploaded or local audio file fails validation."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def validate_extension(filename: str) -> None:
    """Raise ValidationError if the file extension is not supported."""
    ext = Path(filename).suffix.lower()
    if ext not in config.AUDIO_EXTS:
        raise ValidationError(f"Unsupported audio format '{ext}'.", status_code=415)


def validate_file_size(size_bytes: int) -> None:
    """Raise ValidationError if the file exceeds MAX_FILE_SIZE_MB."""
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise ValidationError(
            f"File size {size_mb:.1f} MB exceeds the {config.MAX_FILE_SIZE_MB} MB limit.",
            status_code=413,
        )


def validate_upload(filename: str, size_bytes: int) -> None:
    """Run all validations for an uploaded audio file."""
    if not filename:
        raise ValidationError("Filename must not be empty.", status_code=400)
    validate_extension(filename)
    validate_file_size(size_bytes)
