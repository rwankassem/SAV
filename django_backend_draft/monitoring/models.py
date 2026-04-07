from django.db import models
from trips.models import Trip

class MonitoringStatus(models.Model):
    trip = models.ForeignKey(Trip, on_delete=models.CASCADE, related_name="statuses")
    eye_state = models.CharField(max_length=20, default="unknown")
    yawn_detected = models.BooleanField(default=False)
    head_alert = models.BooleanField(default=False)
    driver_status = models.CharField(max_length=20, default="normal")
    confidence = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.driver_status} - {self.created_at}"