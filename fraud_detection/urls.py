# fraud_detection/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('input/', views.input_view, name='input_view'),
    path('csv_upload/', views.csv_upload_view, name='csv_upload_view'),
]
