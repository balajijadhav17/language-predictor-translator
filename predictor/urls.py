from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('translate/', views.translate, name='translate'),
    path('model-accuracy/', views.model_accuracy, name='model_accuracy'),
]
