from django.urls import path
from .views import predict_text

urlpatterns = [
    path('predict/', predict_text, name='predict')
]
