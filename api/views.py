# api/views.py -- lazy-load artifact version (replace existing file)
import os, traceback, json, time, logging
import numpy as np, pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .serializers import PredictSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(["GET"])
def ping(request):
    # lightweight ping for uptime checks
    return Response({"ping": "pong"})


logger = logging.getLogger("api")

# ARTIFACTS lazy loader
ART_DIR = os.environ.get("ARTIFACTS_DIR", settings.ARTIFACTS_DIR)

_artifacts = None
_art_load_traceback = None
_art_load_lock = False  # simple guard to avoid re-entrancy

def _ensure_artifacts():
    """
    Load artifacts on first use. Returns (artifacts, error_traceback or None).
    This is intentionally lazy to avoid blocking import/startup.
    """
    global _artifacts, _art_load_traceback, _art_load_lock
    if _artifacts is not None or _art_load_traceback is not None:
        return _artifacts, _art_load_traceback

    # Prevent concurrent loads in rare multi-threaded startup
    if _art_load_lock:
        # Another thread is loading; wait briefly and return whatever we have
        time.sleep(0.5)
        return _artifacts, _art_load_traceback

    _art_load_lock = True
    try:
        from .artifact_loader import load_artifacts
        _artifacts = load_artifacts(ART_DIR)
        logger.info("Artifacts loaded lazily OK")
    except Exception:
        _art_load_traceback = traceback.format_exc()
        logger.exception("Lazy artifact load failed")
    finally:
        _art_load_lock = False
    return _artifacts, _art_load_traceback

# Constants for preprocessing
NUM_COLS = ["N", "P", "K", "SOWN", "SOIL_PH", "TEMP", "RELATIVE_HUMIDITY"]
CAT_COLS = ["SOIL"]

def clamp_numeric(field, val, artifacts):
    mm = artifacts.get("minmax", {}).get(field) if artifacts else None
    if mm is None:
        return float(val), False
    v = float(val)
    was = False
    if v < mm["min"]:
        v = mm["min"]; was = True
    elif v > mm["max"]:
        v = mm["max"]; was = True
    return v, was

def preprocess_row(row_dict, artifacts):
    df = pd.DataFrame([row_dict])
    Xnum = df[NUM_COLS].to_numpy(dtype=float)
    Xnum_imp = artifacts["num_imputer"].transform(Xnum)
    Xnum_scaled = artifacts["scaler"].transform(Xnum_imp)
    Xsoil = artifacts["soil_encoder"].transform(df[CAT_COLS].astype(str))
    X = np.hstack([Xnum_scaled, Xsoil]).astype(np.float32)
    return X

def softmax_with_temp(logits, temperature=1.0):
    logits = np.array(logits, dtype=np.float32) / max(1e-6, float(temperature))
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def log_drift(in_row, prediction, artifacts):
    try:
        train_stats = artifacts.get("train_stats", {})
        psi = {}
        for c in NUM_COLS:
            ts = train_stats.get("numeric", {}).get(c)
            if not ts:
                psi[c] = {"error":"no_train_stats"}
                continue
            edges = ts["hist_edges"]
            val = float(in_row[c])
            idx = np.searchsorted(edges, val, side="right") - 1
            idx = int(max(0, min(idx, len(ts["hist_counts"])-1)))
            psi[c] = {"bin_index": idx, "train_count": int(ts["hist_counts"][idx])}
        rec = {"ts": time.time(), "input": in_row, "prediction": prediction, "psi": psi}
        logpath = os.path.join(ART_DIR, "drift_log.jsonl")
        with open(logpath, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        logger.exception("drift log failed")

# Views
class HealthView(APIView):
    def get(self, request):
        # force a full artifact load and return the full traceback string on error
        artifacts, tb = _ensure_artifacts()
        if tb:
            # TEMPORARY DEBUG: return full traceback string so we can see exact error remotely
            return Response({"status": "error", "artifact_traceback": tb}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        try:
            # return artifact keys so we know what loaded
            return Response({"status": "ok", "artifacts_keys": list(artifacts.keys())})
        except Exception:
            # fallback if artifacts is weird
            return Response({"status": "ok", "artifacts_repr": repr(artifacts)})

class PredictView(APIView):
    def post(self, request):
        artifacts, tb = _ensure_artifacts()
        if tb:
            return Response({"error":"server_artifact_load_failed", "detail": "See /api/health for traceback"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        serializer = PredictSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        inp = serializer.validated_data

        clamped = {}
        was_clamped = False
        for c in NUM_COLS:
            v, cl = clamp_numeric(c, inp[c], artifacts)
            inp[c] = v
            if cl:
                clamped[c] = v
                was_clamped = True

        try:
            X = preprocess_row(inp, artifacts)
        except Exception as e:
            logger.exception("preprocess fail")
            return Response({"error":"preprocessing failed", "detail": str(e)}, status=500)

        try:
            interpreter = artifacts["tflite_interpreter"]
            idt = artifacts["tflite_input_details"][0]
            interpreter.set_tensor(idt['index'], X)
            interpreter.invoke()
            outputs = [interpreter.get_tensor(o['index']) for o in artifacts["tflite_output_details"]]
        except Exception as e:
            logger.exception("tflite inference failed")
            return Response({"error":"inference_failed", "detail": str(e)}, status=500)

        crop_logits = outputs[0][0]
        harvested_scaled = float(outputs[1][0][0])
        water_logits = outputs[2][0]
        crop_duration_scaled = float(outputs[3][0][0])

        temp = float(artifacts.get("calibration", {}).get("temperature", 1.0))
        crop_probs = softmax_with_temp(crop_logits, temperature=temp)
        crop_idx = int(np.argmax(crop_probs))
        crop_conf = float(np.max(crop_probs))

        water_probs = softmax_with_temp(water_logits, temperature=1.0)
        water_idx = int(np.argmax(water_probs))

        name = artifacts["name_inv"].get(crop_idx, "unknown")
        water = artifacts["water_inv"].get(water_idx, "unknown")

        inv = artifacts["reg_scaler"].inverse_transform([[harvested_scaled, crop_duration_scaled]])
        harvested_orig = float(inv[0][0])
        crop_duration_orig = float(inv[0][1])

        resp = {
            "NAME": name,
            "HARVESTED": round(harvested_orig, 3),
            "WATER_SOURCE": water,
            "CROP_DURATION": round(crop_duration_orig, 3),
            "CONFIDENCE": round(crop_conf, 4),
            "meta": {"was_clamped": was_clamped, "clamped_fields": clamped}
        }

        try:
            log_drift(inp, resp, artifacts)
        except Exception:
            logger.exception("drift log exception")

        return Response(resp)
