from django.urls import path
from .views import ImageIngestView, ImageSearchView, ImageBatchIngestView

urlpatterns = [
	path('api/images', ImageIngestView.as_view()),
	path('api/images/batch', ImageBatchIngestView.as_view()),
	path('api/search', ImageSearchView.as_view()),
]
