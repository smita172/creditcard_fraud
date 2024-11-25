from django.urls import path
from . import views  # Import views from the app
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home_view, name='home'),
    path('input_prediction/', views.input_prediction, name='input_prediction'),
    path('csv_prediction/', views.upload_csv, name='csv_prediction'),
    path('csv_visualization/', views.csv_visualization, name='csv_visualization'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)