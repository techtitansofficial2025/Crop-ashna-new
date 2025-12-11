from django.urls import path
from .views import PredictView, HealthView, ping, debug_file_list

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("health/", HealthView.as_view(), name="health"),
    path("ping/", ping, name="ping"),
    path("debug_files/", debug_file_list, name="debug_files"),
]
