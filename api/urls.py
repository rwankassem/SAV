from django.urls import path
from .views import latest_status_view, alerts_view

urlpatterns = [
    path("status/", latest_status_view),
    path("alerts/", alerts_view),
]