"""Prometheus metrics middleware for the Flask API."""

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "deepfake_api_requests_total",
        "Total API requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "deepfake_api_request_duration_seconds",
        "API request latency in seconds",
        ["endpoint"],
    )
    PREDICTION_COUNT = Counter(
        "deepfake_predictions_total",
        "Total predictions made",
        ["label"],
    )


def register_metrics(app):
    """Attach Prometheus metrics to a Flask app instance."""
    if not PROMETHEUS_AVAILABLE:
        return

    import time
    from flask import request, Response

    @app.before_request
    def start_timer():
        request._start_time = time.time()

    @app.after_request
    def record_request(response):
        if hasattr(request, "_start_time"):
            latency = time.time() - request._start_time
            REQUEST_LATENCY.labels(endpoint=request.path).observe(latency)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.path,
            status=response.status_code,
        ).inc()
        return response

    @app.route("/metrics", methods=["GET"])
    def metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
