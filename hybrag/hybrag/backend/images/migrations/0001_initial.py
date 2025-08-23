from django.db import migrations, models
import uuid

class Migration(migrations.Migration):

	initial = True

	dependencies = []

	operations = [
		migrations.CreateModel(
			name='ImageItem',
			fields=[
				('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, serialize=False)),
				('file', models.ImageField(upload_to='images/')),
				('building', models.CharField(max_length=128, db_index=True)),
				('shot_date', models.DateField(db_index=True)),
				('notes', models.TextField(blank=True)),
				('created_at', models.DateTimeField(auto_now_add=True)),
			],
		),
	]
