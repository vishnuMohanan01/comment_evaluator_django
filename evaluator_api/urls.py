from django.urls import path
from . import views


app_name = 'evaluator_api'


urlpatterns = [
    path('text/', views.evaluate_text, name='text'),
    path('image/', views.evaluate_image, name='image')
]