import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from web_model import predict_audio_file

app = FastAPI()

# Optional static dir
if not os.path.isdir("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Deepfake Audio Detector</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
            :root {
                --bg: #050816;
                --bg-card: #0f172a;
                --accent: #3b82f6;
                --accent-soft: rgba(59,130,246,0.15);
                --text-main: #e5e7eb;
                --text-muted: #9ca3af;
                --border-subtle: #1f2937;
                --danger: #f97373;
                --success: #4ade80;
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: radial-gradient(circle at top, #1e293b 0, #020617 45%, #000 100%);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 24px;
            }
            .shell {
                width: 100%;
                max-width: 960px;
            }
            .nav {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
            }
            .logo {
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 600;
                letter-spacing: 0.04em;
            }
            .logo-badge {
                width: 32px;
                height: 32px;
                border-radius: 10px;
                background: linear-gradient(135deg, #60a5fa, #a855f7);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
            }
            .nav-pill {
                padding: 4px 10px;
                border-radius: 999px;
                border: 1px solid rgba(148,163,184,0.5);
                font-size: 11px;
                color: var(--text-muted);
            }
            .grid {
                display: grid;
                grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
                gap: 20px;
            }
            @media (max-width: 800px) {
                .grid {
                    grid-template-columns: 1fr;
                }
            }
            .card {
                background: radial-gradient(circle at top left, rgba(148,163,184,0.18), transparent 55%),
                            radial-gradient(circle at bottom right, rgba(59,130,246,0.2), transparent 60%),
                            var(--bg-card);
                border-radius: 18px;
                border: 1px solid rgba(148,163,184,0.3);
                padding: 22px 22px 20px;
                box-shadow: 0 22px 45px rgba(15,23,42,0.9);
                position: relative;
                overflow: hidden;
            }
            .card::before {
                content: "";
                position: absolute;
                inset: 0;
                background: radial-gradient(circle at top, rgba(248,250,252,0.06), transparent 55%);
                opacity: 0.6;
                pointer-events: none;
            }
            .card-inner {
                position: relative;
                z-index: 1;
            }
            .eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 3px 9px;
                border-radius: 999px;
                background: rgba(15,23,42,0.85);
                border: 1px solid rgba(148,163,184,0.65);
                font-size: 11px;
                color: var(--text-muted);
                margin-bottom: 14px;
            }
            .eyebrow-dot {
                width: 6px;
                height: 6px;
                border-radius: 999px;
                background: #22c55e;
                box-shadow: 0 0 0 4px rgba(34,197,94,0.32);
            }
            h1 {
                font-size: 26px;
                line-height: 1.1;
                margin-bottom: 8px;
            }
            .sub {
                font-size: 13px;
                color: var(--text-muted);
                max-width: 460px;
                line-height: 1.4;
                margin-bottom: 22px;
            }
            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 18px;
            }
            .badge {
                font-size: 11px;
                padding: 4px 10px;
                border-radius: 999px;
                border: 1px solid rgba(148,163,184,0.5);
                color: var(--text-muted);
                background: rgba(15,23,42,0.85);
            }
            .badge strong {
                color: #e5e7eb;
            }
            .upload-area {
                border-radius: 16px;
                border: 1px dashed rgba(148,163,184,0.7);
                background: rgba(15,23,42,0.9);
                padding: 16px 16px 14px;
                margin-bottom: 14px;
            }
            .upload-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 10px;
            }
            .upload-title {
                font-size: 13px;
                font-weight: 600;
            }
            .chip {
                font-size: 11px;
                padding: 3px 8px;
                border-radius: 999px;
                background: rgba(15,23,42,0.9);
                border: 1px solid rgba(148,163,184,0.7);
                color: var(--text-muted);
            }
            .upload-main {
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 12px;
                color: var(--text-muted);
            }
            .upload-icon {
                width: 30px;
                height: 30px;
                border-radius: 12px;
                background: radial-gradient(circle at top left, rgba(96,165,250,0.6), rgba(79,70,229,0.9));
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
            }
            .upload-main span {
                display: block;
            }
            .upload-main span strong {
                color: #e5e7eb;
            }
            .upload-main small {
                display: block;
            }
            .upload-actions {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-top: 10px;
                gap: 8px;
            }
            .file-label {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 7px 14px;
                border-radius: 999px;
                background: var(--accent);
                color: white;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                border: none;
            }
            .file-label span {
                font-size: 16px;
            }
            input[type="file"] {
                display: none;
            }
            .filename {
                font-size: 12px;
                color: var(--text-muted);
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                flex: 1;
            }
            .analyse-btn {
                margin-top: 12px;
                padding: 9px 18px;
                border-radius: 999px;
                border: none;
                background: linear-gradient(135deg, #3b82f6, #22c55e);
                color: white;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 14px 30px rgba(15,23,42,0.8);
            }
            .analyse-btn:hover {
                filter: brightness(1.05);
            }
            .hint {
                font-size: 11px;
                color: var(--text-muted);
                margin-top: 6px;
            }

            /* Right card (info) */
            .side-card {
                background: radial-gradient(circle at top right, rgba(59,130,246,0.25), transparent 55%),
                            radial-gradient(circle at bottom left, rgba(34,197,94,0.18), transparent 60%),
                            #020617;
                border-radius: 18px;
                border: 1px solid rgba(30,64,175,0.9);
                padding: 18px 18px 16px;
                position: relative;
                overflow: hidden;
            }
            .side-card::after {
                content: "";
                position: absolute;
                width: 260px;
                height: 260px;
                border-radius: 999px;
                border: 1px solid rgba(148,163,184,0.35);
                top: -130px;
                right: -130px;
                opacity: 0.6;
            }
            .side-inner {
                position: relative;
                z-index: 1;
            }
            .side-title {
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 8px;
            }
            .side-copy {
                font-size: 12px;
                color: #e5e7eb;
                margin-bottom: 12px;
            }
            .metric-row {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 10px;
            }
            .metric {
                flex: 1 1 90px;
                padding: 7px 9px;
                border-radius: 12px;
                background: rgba(15,23,42,0.85);
                border: 1px solid rgba(148,163,184,0.7);
                font-size: 11px;
            }
            .metric-label {
                color: var(--text-muted);
                margin-bottom: 2px;
            }
            .metric-value {
                font-size: 14px;
                font-weight: 600;
            }
            .metric-good {
                color: var(--success);
            }
            .metric-warn {
                color: var(--danger);
            }
            .tagline {
                font-size: 11px;
                color: var(--text-muted);
                margin-top: 8px;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <div class="shell">
            <div class="nav">
                <div class="logo">
                    <div class="logo-badge">DF</div>
                    <div>DeepGuard<span style="opacity:0.6;"> • Audio</span></div>
                </div>
                <div class="nav-pill">Hackathon Build · v1.0</div>
            </div>

            <div class="grid">
                <div class="card">
                    <div class="card-inner">
                        <div class="eyebrow">
                            <div class="eyebrow-dot"></div>
                            Real‑time deepfake screening
                        </div>
                        <h1>Deepfake Audio Detection</h1>
                        <p class="sub">
                            Upload a short audio clip and our model will estimate the likelihood that it was synthetically generated using AI voice cloning.
                        </p>

                        <div class="badge-row">
                            <div class="badge"><strong>110‑D</strong> audio feature vector</div>
                            <div class="badge"><strong>16k+</strong> training samples</div>
                            <div class="badge">Optimized SVM classifier</div>
                        </div>

                        <form action="/predict" enctype="multipart/form-data" method="post" id="uploadForm">
                            <div class="upload-area">
                                <div class="upload-header">
                                    <div class="upload-title">Upload an audio file</div>
                                    <div class="chip">Supported: wav · mp3 · ogg · m4a · flac</div>
                                </div>
                                <div class="upload-main">
                                    <div class="upload-icon">↑</div>
                                    <div>
                                        <span><strong>Drop a file here</strong> or use the button below.</span>
                                        <small>We process everything locally in this demo; clips are not stored after analysis.</small>
                                    </div>
                                </div>
                                <div class="upload-actions">
                                    <label for="file" class="file-label">
                                        <span>📂</span> Choose audio file
                                    </label>
                                    <input id="file" type="file" name="file" accept=".wav,.mp3,.ogg,.m4a,.flac" required />
                                    <div class="filename" id="filename">No file selected</div>
                                </div>
                            </div>

                            <button type="submit" class="analyse-btn">Analyse audio</button>
                            <div class="hint">
                                Tip: try one of your own voice notes vs. an AI‑generated clip to see how the model responds.
                            </div>
                        </form>
                    </div>
                </div>

                <div class="side-card">
                    <div class="side-inner">
                        <div class="side-title">What we look at</div>
                        <p class="side-copy">
                            The detector combines spectral, prosodic, and voice‑print features to capture the subtle artifacts that AI voice models leave behind.
                        </p>
                        <div class="metric-row">
                            <div class="metric">
                                <div class="metric-label">Training AUC</div>
                                <div class="metric-value metric-good">0.99+</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Feature size</div>
                                <div class="metric-value">110 dims</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Focus</div>
                                <div class="metric-value metric-warn">Deepfake speech</div>
                            </div>
                        </div>
                        <p class="tagline">
                            This is a research prototype. Predictions are probabilistic and should be combined with human review, especially for high‑stakes use cases.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const input = document.getElementById("file");
            const nameSpan = document.getElementById("filename");

            input.addEventListener("change", () => {
                if (input.files && input.files.length > 0) {
                    nameSpan.textContent = input.files[0].name;
                } else {
                    nameSpan.textContent = "No file selected";
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    temp_dir = "uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(temp_dir, temp_name)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        label, prob_fake = predict_audio_file(temp_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return f"""
        <html><body>
        <h3>Error processing file: {file.filename}</h3>
        <p>{e}</p>
        <a href="/">Back</a>
        </body></html>
        """

    if os.path.exists(temp_path):
        os.remove(temp_path)

             color = "#0f172a"
    border_color = "#22c55e" if label == "REAL" else "#f97373"
    accent = "#22c55e" if label == "REAL" else "#f97373"
    verdict_text = "Likely human speech" if label == "REAL" else "Suspected deepfake audio"

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Result - Deepfake Audio Detector</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
            body {{
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: radial-gradient(circle at top, #1e293b 0, #020617 45%, #000 100%);
                color: #e5e7eb;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 24px;
            }}
            .shell {{
                width: 100%;
                max-width: 720px;
            }}
            .card {{
                background: {color};
                border-radius: 18px;
                border: 1px solid rgba(148,163,184,0.4);
                padding: 22px 22px 18px;
                box-shadow: 0 22px 45px rgba(15,23,42,0.9);
            }}
            .header-row {{
                display: flex;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 14px;
            }}
            h1 {{
                font-size: 20px;
                margin-bottom: 4px;
            }}
            .sub {{
                font-size: 13px;
                color: #9ca3af;
            }}
            .pill {{
                font-size: 11px;
                padding: 4px 10px;
                border-radius: 999px;
                border: 1px solid rgba(148,163,184,0.5);
                color: #9ca3af;
            }}
            .file-row {{
                margin-top: 8px;
                padding: 10px 12px;
                border-radius: 12px;
                background: rgba(15,23,42,0.9);
                border: 1px solid rgba(148,163,184,0.5);
                font-size: 12px;
            }}
            .file-label {{
                color: #9ca3af;
            }}
            .verdict {{
                margin-top: 14px;
                padding: 12px 14px;
                border-radius: 14px;
                border: 1px solid {border_color};
                background: rgba(15,23,42,0.9);
            }}
            .verdict-title {{
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: {accent};
                margin-bottom: 6px;
            }}
            .verdict-main {{
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 4px;
            }}
            .prob-row {{
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                margin-top: 6px;
            }}
            .prob-bar {{
                margin-top: 8px;
                width: 100%;
                height: 7px;
                border-radius: 999px;
                background: #020617;
                overflow: hidden;
                border: 1px solid rgba(30,64,175,0.8);
            }}
            .prob-fill {{
                height: 100%;
                width: {prob_fake * 100:.1f}%;
                max-width: 100%;
                background: linear-gradient(90deg, #3b82f6, {accent});
            }}
            .footer {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 14px;
                font-size: 11px;
                color: #9ca3af;
            }}
            .btn {{
                padding: 7px 14px;
                border-radius: 999px;
                border: none;
                background: linear-gradient(135deg, #3b82f6, #22c55e);
                color: white;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="shell">
            <div class="card">
                <div class="header-row">
                    <div>
                        <h1>Deepfake analysis complete</h1>
                        <p class="sub">Model evaluation for a single audio clip.</p>
                    </div>
                    <div class="pill">Deepfake Audio Detector · v1.0</div>
                </div>

                <div class="file-row">
                    <span class="file-label">File analysed:</span>
                    <br />
                    <span>{file.filename}</span>
                </div>

                <div class="verdict">
                    <div class="verdict-title">Model verdict</div>
                    <div class="verdict-main">{verdict_text}</div>
                    <div class="prob-row">
                        <span>Predicted label: <strong>{label}</strong></span>
                        <span>Deepfake probability: <strong>{prob_fake:.3f}</strong></span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill"></div>
                    </div>
                </div>

                <div class="footer">
                    <span>Prototype model — use alongside human judgement for critical decisions.</span>
                    <a href="/" class="btn">Analyse another clip</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

