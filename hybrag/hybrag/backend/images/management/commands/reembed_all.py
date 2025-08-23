from __future__ import annotations
from django.core.management.base import BaseCommand
from django.conf import settings
from images.models import ImageItem
from images.views import get_siglip, get_vectors
import os

class Base(BaseCommand):
	pass

class Command(BaseCommand):
	help = "Re-embed existing images with the current model and upsert to Pinecone"

	def add_arguments(self, parser):
		parser.add_argument('--batch', type=int, default=32, help='Embedding batch size')
		parser.add_argument('--reset', action='store_true', help='Delete all existing vectors in index before re-embedding')
		parser.add_argument('--namespace', type=str, default='', help='Optional Pinecone namespace to use')

	def handle(self, *args, **options):
		batch_size = options['batch']
		reset = options['reset']
		namespace = options['namespace'] or None
		qs = ImageItem.objects.all().order_by('created_at')
		total = qs.count()
		if total == 0:
			self.stdout.write(self.style.WARNING('No images to re-embed.'))
			return

		sig = get_siglip()
		vec = get_vectors()

		if reset:
			self.stdout.write(self.style.NOTICE('Deleting all vectors in Pinecone index...'))
			vec.delete_all(namespace=namespace)

		self.stdout.write(self.style.NOTICE(f'Re-embedding {total} images (batch={batch_size})'))

		buf_items = []
		buf_paths = []
		count = 0
		for item in qs.iterator():
			if not os.path.exists(item.file.path):
				continue
			buf_items.append(item)
			buf_paths.append(item.file.path)
			if len(buf_items) >= batch_size:
				self._process_batch(buf_items, buf_paths, sig, vec, namespace)
				count += len(buf_items)
				self.stdout.write(self.style.NOTICE(f'Upserted {count}'))
				buf_items, buf_paths = [], []
		if buf_items:
			self._process_batch(buf_items, buf_paths, sig, vec, namespace)
			count += len(buf_items)
			self.stdout.write(self.style.NOTICE(f'Upserted {count}'))

		self.stdout.write(self.style.SUCCESS('Re-embedding complete.'))

	def _process_batch(self, items, paths, sig, vec, namespace):
		vectors = sig.image_embed_batch(paths)
		upserts = []
		for it, v in zip(items, vectors):
			payload = {
				"id": str(it.id),
				"building": it.building,
				"shot_date": str(it.shot_date),
				"shot_ymd": int(str(it.shot_date).replace('-', '')),
				"image_url": getattr(settings, 'BACKEND_BASE_URL', 'http://127.0.0.1:8000').rstrip('/') + it.file.url,
				"notes": it.notes or "",
			}
			upserts.append({"id": str(it.id), "values": v, "metadata": payload})
		vec.upsert_batch(upserts, namespace=namespace)
