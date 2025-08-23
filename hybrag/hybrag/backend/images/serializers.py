from rest_framework import serializers
from .models import ImageItem

class ImageItemSerializer(serializers.ModelSerializer):
	class Meta:
		model = ImageItem
		fields = ['id', 'file', 'building', 'shot_date', 'notes', 'created_at']
		read_only_fields = ['id', 'created_at']
