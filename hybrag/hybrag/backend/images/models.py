import uuid
from django.db import models

class ImageItem(models.Model):
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	file = models.ImageField(upload_to="images/")
	building = models.CharField(max_length=128, db_index=True)
	shot_date = models.DateField(db_index=True)
	notes = models.TextField(blank=True)
	created_at = models.DateTimeField(auto_now_add=True)

	def __str__(self) -> str:
		return f"{self.id} | {self.building} | {self.shot_date}"
