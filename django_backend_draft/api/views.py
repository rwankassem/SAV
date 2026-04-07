from rest_framework.decorators import api_view
from rest_framework.response import Response
from monitoring.models import MonitoringStatus
from .serializers import MonitoringStatusSerializer


@api_view(["GET"])
def home_view(request):
    return Response({"message": "Drowsiness System API is running"})


@api_view(["GET"])
def latest_status_view(request):
    latest = MonitoringStatus.objects.order_by("-created_at").first()
    if not latest:
        return Response({"message": "No status available"})
    return Response(MonitoringStatusSerializer(latest).data)