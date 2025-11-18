from django.urls import path
from . import views

app_name = 'esp' # Пространство имен приложения (рекомендуется)

urlpatterns = [
    path('receive/', views.receive_data, name='receive_data'),
    path('display/', views.display_data, name='display_data'),
]