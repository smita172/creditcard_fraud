from django.urls import path
from . import views  # Import views from the app

urlpatterns = [
    path('', views.home_view, name='home'),
    path('input_prediction/', views.input_prediction, name='input_prediction'),
]
