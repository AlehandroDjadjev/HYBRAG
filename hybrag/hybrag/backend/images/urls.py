from django.urls import path
from .views import ImageIngestView, ImageSearchView, ImageBatchIngestView, UpsertViaS3View, PresignUploadView, UIIndexView

urlpatterns = [
    path('', UIIndexView.as_view()),
	path('api/images', ImageIngestView.as_view()),
	path('api/images/batch', ImageBatchIngestView.as_view()),
	path('api/images/upsert-s3', UpsertViaS3View.as_view()),
	path('api/presign-upload', PresignUploadView.as_view()),
	path('api/search', ImageSearchView.as_view()),
]
