import os, json, joblib, logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger("artifact_loader")

def load_artifacts(artifacts_dir):
    artifacts = {}
    artifacts_dir = os.path.abspath(artifacts_dir)
    logger.info("Loading artifacts from %s", artifacts_dir)

    # minmax, label maps, calibration, train_stats
    with open(os.path.join(artifacts_dir, "minmax.json")) as f:
        artifacts["minmax"] = json.load(f)
    with open(os.path.join(artifacts_dir, "label_encoders.json")) as f:
        artifacts["label_maps"] = json.load(f)
    with open(os.path.join(artifacts_dir, "calibration.json")) as f:
        artifacts["calibration"] = json.load(f)
    with open(os.path.join(artifacts_dir, "train_stats.json")) as f:
        artifacts["train_stats"] = json.load(f)

    # preprocessors & scalers
    artifacts["num_imputer"] = joblib.load(os.path.join(artifacts_dir, "num_imputer.joblib"))
    artifacts["scaler"] = joblib.load(os.path.join(artifacts_dir, "scaler.joblib"))
    artifacts["soil_encoder"] = joblib.load(os.path.join(artifacts_dir, "soil_encoder.joblib"))

    # regression scaler for inverse transform
    artifacts["reg_scaler"] = joblib.load(os.path.join(artifacts_dir, "reg_scaler.joblib"))

    # TFLite model
    tflite_path = os.path.join(artifacts_dir, "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    artifacts["tflite_interpreter"] = interpreter
    artifacts["tflite_input_details"] = interpreter.get_input_details()
    artifacts["tflite_output_details"] = interpreter.get_output_details()

    # label invert maps
    # label_encoders.json has NAME, NAME_INV etc.
    lmap = artifacts["label_maps"]
    # NAME_INV keys might be strings, convert to int->str mapping
    artifacts["name_inv"] = {int(k): v for k,v in lmap.get("NAME_INV", {}).items()}
    artifacts["water_inv"] = {int(k): v for k,v in lmap.get("WATER_INV", {}).items()}
    artifacts["name_map"] = lmap.get("NAME", {})
    artifacts["water_map"] = lmap.get("WATER_SOURCE", {})

    logger.info("Artifacts loaded successfully")
    return artifacts
