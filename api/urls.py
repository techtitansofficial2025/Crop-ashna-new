from django.urls import path
from .views import PredictView, HealthView

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("health/", HealthView.as_view(), name="health"),
    #path("ping/", ping, name="ping"),
]
