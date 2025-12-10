import os, json, time, logging, numpy as np, pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .serializers import PredictSerializer
from .artifact_loader import load_artifacts

logger = logging.getLogger("api")

ART_DIR = os.environ.get("ARTIFACTS_DIR", settings.ARTIFACTS_DIR)
_artifacts = load_artifacts(ART_DIR)

# helper names
NUM_COLS = ["N","P","K","SOWN","SOIL_PH","TEMP","RELATIVE_HUMIDITY"]
CAT_COLS = ["SOIL"]

def clamp_numeric(field, val):
    mm = _artifacts["minmax"].get(field)
    if mm is None:
        return float(val), False
    v = float(val)
    was = False
    if v < mm["min"]:
        v = mm["min"]; was = True
    elif v > mm["max"]:
        v = mm["max"]; was = True
    return v, was

def preprocess_row(row_dict):
    # row_dict keys are NUM_COLS + CAT_COLS
    df = pd.DataFrame([row_dict])
    # numeric impute & scale
    Xnum = df[NUM_COLS].to_numpy(dtype=float)
    Xnum_imp = _artifacts["num_imputer"].transform(Xnum)
    Xnum_scaled = _artifacts["scaler"].transform(Xnum_imp)
    Xsoil = _artifacts["soil_encoder"].transform(df[CAT_COLS].astype(str))
    X = np.hstack([Xnum_scaled, Xsoil]).astype(np.float32)
    return X

def softmax_with_temp(logits, temperature=1.0):
    logits = np.array(logits, dtype=np.float32) / max(1e-6, float(temperature))
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def log_drift(in_row, prediction):
    # simple PSI-like logging; store record line-by-line
    try:
        train_stats = _artifacts["train_stats"]
        psi = {}
        for c in NUM_COLS:
            ts = train_stats["numeric"].get(c)
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

class HealthView(APIView):
    def get(self, request):
        return Response({"status":"ok"})

class PredictView(APIView):
    def post(self, request):
        serializer = PredictSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        inp = serializer.validated_data

        # Clamp numeric values to train min/max (auto-clamp behaviour)
        clamped = {}
        was_clamped = False
        for c in NUM_COLS:
            v, cl = clamp_numeric(c, inp[c])
            inp[c] = v
            if cl:
                clamped[c] = v
                was_clamped = True

        # preprocess
        try:
            X = preprocess_row(inp)
        except Exception as e:
            logger.exception("preprocess fail")
            return Response({"error":"preprocessing failed", "detail": str(e)}, status=500)

        # TFLite inference
        interpreter = _artifacts["tflite_interpreter"]
        idt = _artifacts["tflite_input_details"][0]
        interpreter.set_tensor(idt['index'], X)
        interpreter.invoke()
        outputs = [interpreter.get_tensor(o['index']) for o in _artifacts["tflite_output_details"]]

        # mapping: [crop_logits, harvested, water_logits, crop_duration]
        crop_logits = outputs[0][0]
        harvested_scaled = float(outputs[1][0][0])
        water_logits = outputs[2][0]
        crop_duration_scaled = float(outputs[3][0][0])

        # apply temperature scaling
        temp = float(_artifacts.get("calibration", {}).get("temperature", 1.0))
        crop_probs = softmax_with_temp(crop_logits, temperature=temp)
        crop_idx = int(np.argmax(crop_probs))
        crop_conf = float(np.max(crop_probs))

        water_probs = softmax_with_temp(water_logits, temperature=1.0)
        water_idx = int(np.argmax(water_probs))

        name = _artifacts["name_inv"].get(crop_idx, "unknown")
        water = _artifacts["water_inv"].get(water_idx, "unknown")

        # inverse-transform regression scalers
        inv = _artifacts["reg_scaler"].inverse_transform([[harvested_scaled, crop_duration_scaled]])
        harvested_orig = float(inv[0][0])
        crop_duration_orig = float(inv[0][1])

        resp = {
            "NAME": name,
            "HARVESTED": round(harvested_orig, 3),
            "WATER_SOURCE": water,
            "CROP_DURATION": round(crop_duration_orig, 3),
            "CONFIDENCE": round(crop_conf, 4),
            "meta": {
                "was_clamped": was_clamped,
                "clamped_fields": clamped
            }
        }

        # log drift (best-effort, non-blocking small write)
        try:
            log_drift(inp, resp)
        except Exception:
            logger.exception("drift log exception")

        return Response(resp)
