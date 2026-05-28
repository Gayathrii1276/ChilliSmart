from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np
from PIL import Image
import io
import os
import time

AGMARKNET_RESOURCE_ID = "6141ea17-a69d-4713-b600-0a43c8fd9a6c"
AGMARKNET_DEFAULT_URL = f"https://api.data.gov.in/resource/{AGMARKNET_RESOURCE_ID}"
DEFAULT_AGMARKNET_KEY = "579b464db66ec23bdd00000119c3b67fb37c4ee14b0cc2dfcb6795bd"

app = FastAPI(title="MirchiVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static files if present
if os.path.isdir(os.path.join(os.path.dirname(__file__), "static")):
    try:
        from fastapi.staticfiles import StaticFiles
        app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
    except Exception:
        pass


@app.get("/")
def root():
    # serve the main frontend file at root for PWAs
    path = os.path.join(os.path.dirname(__file__), "mirchi_quality_analyzer.html")
    if os.path.isfile(path):
        return FileResponse(path, media_type="text/html")
    return {"status": "ok"}

# lazy-loaded CNN model (optional)
_cnn_model = None


def _load_cnn_if_needed():
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model
    try:
        from tensorflow.keras.models import load_model
        model_path = os.path.join(os.path.dirname(__file__), "mirchi_model.h5")
        if os.path.isfile(model_path):
            _cnn_model = load_model(model_path)
        else:
            _cnn_model = None
    except Exception:
        _cnn_model = None
    return _cnn_model


def pil_to_bgr_array(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)  # RGB
    # convert to BGR like OpenCV
    return arr[:, :, ::-1]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/mirchi_quality_analyzer.html")
def frontend():
    path = os.path.join(os.path.dirname(__file__), "mirchi_quality_analyzer.html")
    if os.path.isfile(path):
        return FileResponse(path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/manifest.json")
def manifest():
    path = os.path.join(os.path.dirname(__file__), "manifest.json")
    if os.path.isfile(path):
        return FileResponse(path, media_type="application/json")
    raise HTTPException(status_code=404, detail="Manifest not found")


@app.get("/service-worker.js")
def sw():
    path = os.path.join(os.path.dirname(__file__), "service-worker.js")
    if os.path.isfile(path):
        return FileResponse(path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = pil_to_bgr_array(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    h, w = img.shape[:2]
    # basic color stats
    mean_bgr = img.mean(axis=(0, 1)).tolist()  # B, G, R
    mean_rgb = mean_bgr[::-1]

    # brightness (luma approximation)
    brightness = (0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2])

    # edge / sharpness via laplacian variance
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())
    except Exception:
        lap_var = None

    # simple defect proxy: percent of very dark pixels
    gray_simple = np.dot(img[..., ::-1], [0.2126, 0.7152, 0.0722])
    dark_ratio = float((gray_simple < 30).sum() / (h * w))

    result = {
        "width": int(w),
        "height": int(h),
        "mean_rgb": [float(v) for v in mean_rgb],
        "brightness": float(brightness),
        "laplacian_variance": lap_var,
        "dark_ratio": dark_ratio,
        "model_available": bool(_load_cnn_if_needed()),
    }

    # if model available, run prediction (best-effort)
    model = _load_cnn_if_needed()
    if model is not None:
        try:
            # resize to model input if possible
            import tensorflow as tf
            input_shape = model.input_shape
            target_h = input_shape[1] or 224
            target_w = input_shape[2] or 224
            img_rgb = img[:, :, ::-1]
            img_resized = tf.image.resize(img_rgb, [target_h, target_w]) / 255.0
            preds = model.predict(np.expand_dims(img_resized.numpy(), 0))
            result["model_prediction"] = preds.tolist()
        except Exception:
            result["model_prediction"] = None

    return JSONResponse(result)


@app.get("/mandi-prices")
def mandi_prices(state: Optional[str] = None, commodity: Optional[str] = "chili", limit: Optional[int] = 50, key: Optional[str] = None):
    """Return mandi/market price entries expected by the frontend.

    Query params supported: `state`, `commodity`, `limit`, `key`.
    If `AGROMARKET_API_URL` or `MANDI_API_URL` is set in environment, this will
    attempt to proxy/translate the request to that API and return an array of
    mandi records. On any failure, a local mock dataset is returned.
    """
    MANDI_API_URL = os.environ.get("AGROMARKET_API_URL") or os.environ.get("MANDI_API_URL") or AGMARKNET_DEFAULT_URL
    AGMARK_KEY = key or os.environ.get("AGMARKNET_KEY") or os.environ.get("AGMARKET_KEY") or DEFAULT_AGMARKNET_KEY

    def _parse_json_resp(text: str):
        import json
        try:
            obj = json.loads(text)
        except Exception:
            return None
        # common wrappers: data / records / results
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("data", "records", "results", "prices", "market_data"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
            # sometimes API returns {status:..., result: {records: [...]}}
            for v in obj.values():
                if isinstance(v, list):
                    return v
        return None

    if MANDI_API_URL:
        try:
            from urllib.parse import urlencode
            import urllib.request

            params = {
                "api-key": AGMARK_KEY,
                "format": "json",
                "limit": limit,
                "filters[state]": state or "",
                "filters[commodity]": commodity or "",
            }
            url = MANDI_API_URL
            if "?" in url:
                url += "&" + urlencode({k: v for k, v in params.items() if v is not None and v != ""})
            else:
                url += "?" + urlencode({k: v for k, v in params.items() if v is not None and v != ""})
            with urllib.request.urlopen(url, timeout=8) as resp:
                text = resp.read().decode("utf-8")
                parsed = _parse_json_resp(text)
                if parsed is not None:
                    normalized = []
                    for row in parsed:
                        if not isinstance(row, dict):
                            continue
                        normalized.append({
                            "market": row.get("market") or row.get("market_name") or row.get("marketname") or row.get("market_name_en") or row.get("market_name_hi") or "—",
                            "district": row.get("district") or row.get("district_name") or row.get("district_name_en") or "",
                            "state": row.get("state") or row.get("state_name") or state or "",
                            "commodity": row.get("commodity") or row.get("commodity_name") or commodity or "",
                            "variety": row.get("variety") or row.get("variety_name") or row.get("variety_name_en") or "Red Dry",
                            "min_price": row.get("min_price") or row.get("minimum_price") or row.get("min_price_rs") or row.get("min"),
                            "max_price": row.get("max_price") or row.get("maximum_price") or row.get("max_price_rs") or row.get("max"),
                            "modal_price": row.get("modal_price") or row.get("modal_price_rs") or row.get("modal"),
                            "arrival_date": row.get("arrival_date") or row.get("date") or row.get("arrival_date1") or row.get("date_of_arrival") or "",
                        })
                    if normalized:
                        return JSONResponse(normalized[: int(limit or 50)])
        except Exception:
            # best-effort; fall back to mock data
            pass

    # Fallback mock data generator matching frontend expectations
    from datetime import datetime, timedelta
    today = datetime.utcnow()
    markets = [
        ("Guntur", "Guntur"), ("Warangal", "Warangal"), ("Khammam", "Khammam"),
        ("Nizamabad", "Nizamabad"), ("Karimnagar", "Karimnagar"), ("Hyderabad", "Hyderabad")
    ]
    rows = []
    for i in range(min(10, max(1, int(limit or 10)))):
        d = today - timedelta(days=i)
        arrival_date = d.strftime("%d/%m/%Y")
        for m, dist in markets:
            base = 140 + (i * -2) + (hash(m) % 12 - 6)
            rows.append({
                "market": m,
                "district": dist,
                "state": state or "",
                "commodity": commodity,
                "variety": "Red Dry",
                "min_price": max(50, base - 18),
                "max_price": base + 28,
                "modal_price": base,
                "arrival_date": arrival_date,
            })
    return JSONResponse(rows)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
