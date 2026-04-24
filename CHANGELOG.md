# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `config.py` — centralised, environment-variable-overridable configuration
- `augment.py` — Gaussian noise, time-stretch, pitch-shift and random-crop helpers
- `extra_features.py` — ZCR, RMS energy, Tonnetz and Mel-spectrogram statistics
- `validation.py` — reusable file-extension and size-limit validators
- `cache.py` — thread-safe LRU prediction cache keyed on SHA-256
- `preprocess.py` — ffmpeg-based audio format normalisation to 16 kHz WAV
- `monitoring.py` — Prometheus metrics middleware (`/metrics` endpoint)
- `feature_importance.py` — permutation-importance analysis with named features
- `batch_predict.py` — directory-level batch classification (CSV / JSON output)
- `sliding_predict.py` — sliding-window inference for long audio files
- `model_registry.py` — versioned model artefact management with metadata
- `evaluate.py` — post-training evaluation script with confusion-matrix PNG
- `tune_model.py` — GridSearchCV hyperparameter optimisation
- `train_rf.py` — RandomForest baseline trainer
- `compare_models.py` — SVM vs RF vs GradientBoosting benchmark
- Flask API: `/info`, `/predict/sliding`, `/metrics` endpoints
- `tests/` — full unit and integration test suite (features, API, augmentation,
  cache, validation, preprocess, sliding predict, model registry)
- GitHub Actions CI: lint + multi-Python matrix, Docker build workflow
- `Makefile` — `install`, `test`, `lint`, `format`, `train`, `docker-*` targets
- `CONTRIBUTING.md`, `LICENSE`, `pyproject.toml`

### Changed
- `extract_features.py` refactored to support augmentation and unified waveform pipeline
- `train_model.py` now uses `StandardScaler + SVC` pipeline, 5-fold CV, ROC-AUC
- `app.py` hardened with file-size guard, format validation, structured logging
- `predict.py` rewritten as argparse CLI with `--verbose` flag
- `Dockerfile` converted to multi-stage build
- `requirements.txt` pinned with minimum versions; `soundfile` added

---

## [0.1.0] — 2026-04-24

### Added
- Initial release: MFCC feature extraction, SVM classifier, Flask API, Docker support
