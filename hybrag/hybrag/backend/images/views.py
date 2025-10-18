from __future__ import annotations
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.views.generic import TemplateView
from urllib.parse import urlparse, unquote
from .models import ImageItem

# S3 helpers
from storage.s3 import presign_put, presign_get

_siglip = None
_vectors = None
_spell = None

DOMAIN_SYNONYMS = {
	"excavator": ["excavator", "digger", "backhoe", "construction excavator"],
	"bulldozer": ["bulldozer", "dozer"],
	"crane": ["crane", "tower crane", "mobile crane"],
	"brick": ["brick", "masonry"],
	"cable": ["cable", "wire", "electrical cable"],
}


def get_siglip():
	global _siglip
	if _siglip is None:
		from .embeddings.siglip import SiglipService
		_siglip = SiglipService(settings.SIGLIP_MODEL_NAME, settings.DEVICE)
	return _siglip


def get_vectors():
	global _vectors
	if _vectors is None:
		try:
			from .vector.s3_vector_store import S3VectorStore
			_vectors = S3VectorStore(
				bucket=getattr(settings, 'VECTOR_S3_BUCKET', None),
				index_name=getattr(settings, 'VECTOR_S3_INDEX', 'images'),
				dim=getattr(settings, 'OS_EMB_DIM', 1536),
				prefix=getattr(settings, 'VECTOR_S3_PREFIX', 'vectors/'),
				region=getattr(settings, 'AWS_REGION', None),
			)
		except Exception:
			# Fallback: try OpenSearch if S3 setup fails
			from .vector.pinecone_store import VectorStore
			_vectors = VectorStore(
				api_key='', environment=None, index_name=getattr(settings, 'OS_INDEX', 'media-embeddings'), dim=getattr(settings, 'OS_EMB_DIM', 1536), host=getattr(settings, 'OS_HOST', None),
			)
	return _vectors


def absolute_media_url(rel_url: str) -> str:
	base = getattr(settings, 'BACKEND_BASE_URL', '')
	if rel_url.startswith('http://') or rel_url.startswith('https://'):
		return rel_url
	return f"{base.rstrip('/')}" + rel_url


def normalize_text_query(q: str) -> str:
	q = (q or '').strip().lower()
	if not q:
		return q
	try:
		from spellchecker import SpellChecker
		global _spell
		if _spell is None:
			_spell = SpellChecker()
			for w in ("cable", "resistor", "conduit", "brick", "excavator", "bulldozer", "crane"):
				_spell.word_frequency.add(w)
		words = q.split()
		corrected = []
		for w in words:
			cw = _spell.correction(w) or w
			corrected.append(cw)
		return ' '.join(corrected)
	except Exception:
		return q


def rerank_with_metadata_boost(results, building: str | None):
	if not results:
		return results
	boosted = []
	for r in results:
		score = r.get('score', 0.0)
		if building and r.get('building') == building:
			score += 0.02
		boosted.append((score, r))
	boosted.sort(key=lambda x: x[0], reverse=True)
	return [r for _, r in boosted]


class PresignUploadView(APIView):
	permission_classes = [permissions.AllowAny]

	def post(self, request):
		key = request.data.get('key')
		content_type = request.data.get('content_type', 'image/jpeg')
		if not key:
			return Response({"detail": "key required"}, status=400)
		return Response(presign_put(key, content_type))


class ImageIngestView(APIView):
	permission_classes = [permissions.AllowAny]

	def post(self, request):
		uploaded = request.FILES.get('file')
		building = request.data.get('building')
		shot_date = request.data.get('shot_date')
		notes = request.data.get('notes', '')

		if not (uploaded and building and shot_date):
			return Response({"detail": "file, building, shot_date required"}, status=status.HTTP_400_BAD_REQUEST)

		return Response({"detail": "Direct file uploads are deprecated. Use S3 presign flow and /api/images/upsert-s3."}, status=400)


class ImageBatchIngestView(APIView):
	permission_classes = [permissions.AllowAny]

	def post(self, request):
		return Response({"detail": "Batch local ingest deprecated. Use S3 presign + upsert-s3 per image."}, status=400)


class UpsertViaS3View(APIView):
	permission_classes = [permissions.AllowAny]

	def post(self, request):
		item_id = request.data.get('id')
		s3_key = request.data.get('s3_key')
		building = request.data.get('building')
		shot_date = request.data.get('shot_date')
		notes = request.data.get('notes', '')
		if not (item_id and s3_key and building and shot_date):
			return Response({"detail": "id, s3_key, building, shot_date required"}, status=400)

		# Generate embedding using a fresh presigned URL, but store only stable s3_key in the index
		img_url = presign_get(s3_key)
		vec = get_siglip()._invoke({"image_url": img_url, "normalize": True})
		payload = {
			"id": str(item_id),
			"building": building,
			"shot_date": str(shot_date),
			"shot_ymd": int(str(shot_date).replace('-', '')) if isinstance(shot_date, str) else 0,
			"s3_key": s3_key,
			"notes": notes or "",
		}
		get_vectors().upsert(str(item_id), vec, payload, namespace=None)
		return Response({"id": str(item_id), "image_url": img_url, "s3_key": s3_key}, status=201)


class ImageSearchView(APIView):
	permission_classes = [permissions.AllowAny]

	def get(self, request):
		q = request.query_params.get('q')
		query_image_id = request.query_params.get('query_image_id')
		building = request.query_params.get('building')
		date_from = request.query_params.get('date_from')
		date_to = request.query_params.get('date_to')
		namespace = request.query_params.get('namespace') or (getattr(settings, 'PINECONE_NAMESPACE', '') or None)
		try:
			top_k = int(request.query_params.get('k', 10))
		except Exception:
			top_k = 10

		if q:
			q_norm = normalize_text_query(q)
			terms = DOMAIN_SYNONYMS.get(q_norm, [q_norm])
			import numpy as np
			# Sequential to reduce contention/timeouts with async endpoint
			embs = [get_siglip().text_embed(t) for t in terms]
			query_vec = np.mean(np.array(embs, dtype=float), axis=0).tolist()
		elif query_image_id:
			return Response({"detail": "Provide a query image via S3 key flow; local files not supported"}, status=400)
		else:
			return Response({"detail": "provide q or query_image_id"}, status=status.HTTP_400_BAD_REQUEST)

		results = get_vectors().search(
			query_vec, top_k=top_k, building=building, date_from=date_from, date_to=date_to, namespace=namespace
		)
		results = rerank_with_metadata_boost(results, building)
		# Attach fresh presigned URLs for any result that has s3_key
		for r in results:
			try:
				s3k = r.get('s3_key')
				if s3k:
					r['image_url'] = presign_get(s3k)
				else:
					# Backward compatibility: derive s3_key from legacy image_url if it points to our bucket
					img = r.get('image_url')
					if img and isinstance(img, str):
						u = urlparse(img)
						host = (u.hostname or '').lower()
						path = unquote(u.path or '')
						bucket = getattr(settings, 'S3_BUCKET', '')
						if bucket and bucket.lower() in host:
							# virtual-hosted: <bucket>.s3[.region].amazonaws.com/<key>
							s3_key = path.lstrip('/')
							if s3_key:
								r['image_url'] = presign_get(s3_key)
			except Exception:
				pass
		return Response({"results": results, "namespace": namespace or ''})


class UIIndexView(TemplateView):
	template_name = 'images/ui.html'
