from django.urls import path
from . import views

app_name = 'esp'  # Пространство имен приложения (рекомендуется)

urlpatterns = [
    path('receive/', views.receive_data, name='receive_data'),
    path('analyze/', views.analyze_overview, name='analyze_overview'),
    path('flight/<int:flight_id>/', views.flight_detail, name='flight_detail'),
    path('delete/<int:flight_id>/', views.delete_flight, name='delete_flight'),
]
