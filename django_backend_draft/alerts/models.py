from django.db import models
from trips.models import Trip

class Alert(models.Model):
    ALERT_TYPES = [
        ("eye", "Eye Closure"),
        ("yawn", "Yawn"),
        ("head", "Head Pose"),
        ("drowsy", "Drowsiness"),
    ]

    trip = models.ForeignKey(Trip, on_delete=models.CASCADE, related_name="alerts")
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    severity = models.CharField(max_length=20, default="high")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.alert_type} - {self.created_at}"