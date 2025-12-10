import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET", "change-this-in-prod")

DEBUG = os.environ.get("DEBUG", "false").lower() in ("1","true","yes")

ALLOWED_HOSTS = ["*"]  # lock this down in production

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

# Simple logging to stdout
LOGGING = {
    "version": 1,
    "handlers": {"console": {"class":"logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}

# Artifacts dir - override with env var
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", str(BASE_DIR + "/../model_artifacts"))
