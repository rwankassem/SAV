from rest_framework import serializers
from monitoring.models import MonitoringStatus
from alerts.models import Alert

class MonitoringStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonitoringStatus
        fields = "__all__"

class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = "__all__"