from django.db import models
from drivers.models import Driver

class Trip(models.Model):
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE, related_name="trips")
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(blank=True, null=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Trip {self.id} - {self.driver.name}"