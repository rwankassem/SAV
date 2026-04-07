from django.db import models

class Driver(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    vehicle_number = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return self.name