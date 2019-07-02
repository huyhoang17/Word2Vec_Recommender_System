from django.urls import path
from api.views.recommender_view import RecommenderView


app_name = 'luxstay_api'

urlpatterns = [
    path('luxstay', RecommenderView.as_view(), name='luxstay')
]
