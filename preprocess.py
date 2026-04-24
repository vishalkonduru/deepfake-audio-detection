"""Pre-processing pipeline: convert non-WAV formats to WAV before inference."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def to_wav(input_path: str, target_sr: int = 16000) -> str:
    """Convert *input_path* to a 16-bit mono WAV at *target_sr* Hz.

    Returns the path to the temporary WAV file (caller must delete it).
    Raises RuntimeError if ffmpeg is unavailable for non-WAV inputs.
    """
    ext = Path(input_path).suffix.lower()
    if ext == ".wav":
        return input_path  # already WAV – no conversion needed

    if not _ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required to convert non-WAV audio but was not found on PATH."
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(target_sr),
        "-ac", "1",
        "-sample_fmt", "s16",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(tmp.name)
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")

    log.info("Converted %s -> %s", input_path, tmp.name)
    return tmp.name
