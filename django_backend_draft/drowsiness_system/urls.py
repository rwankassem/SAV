from django.contrib import admin
from django.urls import path, include
from api.views import home_view

urlpatterns = [
    path("", home_view),
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
]