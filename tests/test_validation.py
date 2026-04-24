"""Unit tests for validation helpers."""

import pytest
from validation import ValidationError, validate_extension, validate_file_size, validate_upload
import config


class TestValidateExtension:
    def test_wav_passes(self):
        validate_extension("audio.wav")  # no exception

    def test_flac_passes(self):
        validate_extension("clip.flac")

    def test_mp3_passes(self):
        validate_extension("song.mp3")

    def test_txt_raises_415(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_extension("document.txt")
        assert exc_info.value.status_code == 415

    def test_pdf_raises_415(self):
        with pytest.raises(ValidationError):
            validate_extension("report.pdf")

    def test_case_insensitive(self):
        validate_extension("audio.WAV")  # should not raise


class TestValidateFileSize:
    def test_small_file_passes(self):
        validate_file_size(1024 * 1024)  # 1 MB

    def test_exact_limit_passes(self):
        validate_file_size(config.MAX_FILE_SIZE_MB * 1024 * 1024)

    def test_over_limit_raises_413(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_file_size((config.MAX_FILE_SIZE_MB + 1) * 1024 * 1024)
        assert exc_info.value.status_code == 413


class TestValidateUpload:
    def test_empty_filename_raises_400(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_upload("", 1024)
        assert exc_info.value.status_code == 400

    def test_valid_upload_passes(self):
        validate_upload("test.wav", 1024 * 1024)

    def test_bad_extension_raises(self):
        with pytest.raises(ValidationError):
            validate_upload("test.exe", 1024)
