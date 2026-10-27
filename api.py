from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
app = FastAPI(title="Quantix MA Engine Local API", version="0.1.0", description="Local research endpoints. Run bound to 127.0.0.1.")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class MatchFeatures(BaseModel):
    features: dict[str, float] = Field(..., description="Pre-bout numeric features named exactly as the trained feature_ columns.")


@app.get("/health")
def health():
    return {"status": "ok", "mode": "local research API", "medical_or_betting_advice": False}


@app.post("/telemetry/summary")
async def telemetry_summary(file: UploadFile = File(...)):
    """Summarize an authorized telemetry CSV; uploaded bytes are deleted after the request."""
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Upload a CSV telemetry file")
    try:
        with TemporaryDirectory(prefix="quantix_telemetry_", dir=ROOT / "outputs") as tmp:
            source = Path(tmp) / "input.csv"
            total = 0
            with source.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="CSV exceeds 200 MiB limit")
                    handle.write(chunk)
            from martial_arts_engine.wearable_telemetry import summarize_telemetry
            summary = summarize_telemetry(source, Path(tmp) / "summary.csv")
            return {"rows": summary.to_dict(orient="records"), "interpretation": "Descriptive trace only; no concussion, injury, lactate or career-risk inference."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        await file.close()


@app.post("/matches/predict")
def match_predict(request: MatchFeatures):
    model_dir = ROOT / "models" / "match_predictor"
    if not (model_dir / "model.joblib").is_file():
        raise HTTPException(status_code=409, detail="No trained model found. Train with verified, licensed, event-grouped data first.")
    try:
        from martial_arts_engine.match_prediction import predict_match
        return predict_match(model_dir, request.features)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from exc
