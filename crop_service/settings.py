import os
from pathlib import Path

# ---------------------------------------------------------
# Base directory
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# SECURITY
# ---------------------------------------------------------
SECRET_KEY = os.environ.get("DJANGO_SECRET", "replace-this-for-production")
DEBUG = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")

ALLOWED_HOSTS = ["*"]  # tighten later if needed

# ---------------------------------------------------------
# Installed apps
# ---------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "rest_framework",
    "api",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "crop_service.urls"

TEMPLATES = []

WSGI_APPLICATION = "crop_service.wsgi.application"

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
LOGGING = {
    "version": 1,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}

# ---------------------------------------------------------
# ARTIFACTS DIRECTORY FIXED
# ---------------------------------------------------------

# Render environment variable overrides everything
_artifacts_env = os.environ.get("ARTIFACTS_DIR")

if _artifacts_env:
    # Allow absolute or relative paths
    ARTIFACTS_DIR = Path(_artifacts_env).resolve()
else:
    # Default: project root / model_artifacts
    ARTIFACTS_DIR = (BASE_DIR / "model_artifacts")

# Always provide string version for code that expects a str
ARTIFACTS_DIR = str(ARTIFACTS_DIR)

# Optional safety check:
if not os.path.isdir(ARTIFACTS_DIR):
    raise RuntimeError(
        f"ARTIFACTS_DIR not found: {ARTIFACTS_DIR}. "
        f"Ensure model_artifacts/ is included or set ARTIFACTS_DIR env var correctly."
    )

# ---------------------------------------------------------
# Static files (Render serves none automatically)
# ---------------------------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
