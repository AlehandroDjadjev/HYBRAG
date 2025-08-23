from __future__ import annotations
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from datasets import load_dataset
from images.models import ImageItem
from images.views import get_siglip, get_vectors
from urllib.parse import urlparse
import os
import requests
import uuid
from itertools import islice

class Command(BaseCommand):
	help = "Ingest images from a Hugging Face dataset and upsert to Pinecone"

	def add_arguments(self, parser):
		parser.add_argument('--dataset', type=str, required=False,
			help='HF dataset path', default='Nexdata/58255_Images_Object_Detection_Data_in_Construction_Site_Scenes')
		parser.add_argument('--split', type=str, default='train', help='Dataset split to load')
		parser.add_argument('--limit', type=int, default=200, help='Max records to ingest')
		parser.add_argument('--building', type=str, default='Dataset', help='Default building metadata')
		parser.add_argument('--date', type=str, default='2024-01-01', help='Default shot_date')
		parser.add_argument('--streaming', action='store_true', help='Use streaming mode to avoid Arrow writes')
		parser.add_argument('--cache_dir', type=str, default='', help='Optional cache dir for HF datasets')

	def handle(self, *args, **options):
		ds_name = options['dataset']
		split = options['split']
		limit = options['limit']
		default_building = options['building']
		default_date = options['date']
		use_streaming = options['streaming']
		cache_dir = options['cache_dir'] or None

		self.stdout.write(self.style.NOTICE(
			f"Loading dataset {ds_name}:{split} (limit={limit}) streaming={use_streaming}"
		))

		kwargs = {}
		if cache_dir:
			os.makedirs(cache_dir, exist_ok=True)
			kwargs['cache_dir'] = cache_dir

		if use_streaming:
			ds = load_dataset(ds_name, split=split, streaming=True, **kwargs)
			iterator = islice(ds, limit)
		else:
			ds = load_dataset(ds_name, split=split, **kwargs)
			iterator = islice(ds, limit)

		media_root = settings.MEDIA_ROOT if hasattr(settings, 'MEDIA_ROOT') else 'media'
		os.makedirs(os.path.join(media_root, 'images'), exist_ok=True)

		items = []
		paths = []
		count = 0
		for ex in iterator:
			# dataset can expose different image fields; try common keys
			image = None
			for key in ('image', 'image_path', 'img', 'file'):
				if key in ex and ex[key] is not None:
					image = ex[key]
					break
			if image is None:
				continue

			local_path = None
			try:
				if hasattr(image, 'save'):
					fname = f"hf_{uuid.uuid4().hex}.jpg"
					local_path = os.path.join(media_root, 'images', fname)
					image.save(local_path)
				elif isinstance(image, str) and (image.startswith('http://') or image.startswith('https://')):
					fname = os.path.basename(urlparse(image).path) or f"hf_{uuid.uuid4().hex}.jpg"
					local_path = os.path.join(media_root, 'images', fname)
					r = requests.get(image, timeout=30)
					r.raise_for_status()
					with open(local_path, 'wb') as f:
						f.write(r.content)
				elif isinstance(image, str) and os.path.exists(image):
					fname = f"hf_{uuid.uuid4().hex}{os.path.splitext(image)[1] or '.jpg'}"
					local_path = os.path.join(media_root, 'images', fname)
					with open(image, 'rb') as src, open(local_path, 'wb') as dst:
						dst.write(src.read())
				else:
					continue
			except Exception:
				continue

			item = ImageItem.objects.create(
				file=os.path.relpath(local_path, media_root).replace('\\', '/'),
				building=default_building,
				shot_date=default_date,
				notes='hf-dataset'
			)
			items.append(item)
			paths.append(os.path.join(media_root, item.file.name if hasattr(item.file, 'name') else item.file))
			count += 1

		if not items:
			raise CommandError('No images ingested from dataset')

		self.stdout.write(self.style.NOTICE(f"Embedding {len(items)} images..."))
		vecs = get_siglip().image_embed_batch(paths)

		self.stdout.write(self.style.NOTICE("Upserting to Pinecone..."))
		upserts = []
		for it, vec in zip(items, vecs):
			payload = {
				"id": str(it.id),
				"building": it.building,
				"shot_date": str(it.shot_date),
				"shot_ymd": int(str(it.shot_date).replace('-', '')),
				"image_url": f"/media/{it.file}",
				"notes": it.notes or "",
			}
			upserts.append({"id": str(it.id), "values": vec, "metadata": payload})

		get_vectors().upsert_batch(upserts)
		self.stdout.write(self.style.SUCCESS(f"Ingested and upserted {len(items)} images."))
