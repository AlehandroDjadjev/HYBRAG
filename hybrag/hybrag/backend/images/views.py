from __future__ import annotations
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.conf import settings
from django.shortcuts import get_object_or_404
from .models import ImageItem

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
		if not settings.PINECONE_API_KEY:
			raise RuntimeError('PINECONE_API_KEY not set')
		from .vector.pinecone_store import VectorStore
		svc = get_siglip()
		_vectors = VectorStore(
			api_key=settings.PINECONE_API_KEY,
			environment=settings.PINECONE_ENV,
			index_name=settings.PINECONE_INDEX,
			dim=svc.dim or 512,
			host=settings.PINECONE_HOST,
		)
	return _vectors


def absolute_media_url(rel_url: str) -> str:
	base = getattr(settings, 'BACKEND_BASE_URL', '')
	if rel_url.startswith('http://') or rel_url.startswith('https://'):
		return rel_url
	return f"{base.rstrip('/')}{rel_url}"


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


class ImageIngestView(APIView):
	permission_classes = [permissions.AllowAny]

	def post(self, request):
		uploaded = request.FILES.get('file')
		building = request.data.get('building')
		shot_date = request.data.get('shot_date')
		notes = request.data.get('notes', '')

		if not (uploaded and building and shot_date):
			return Response({"detail": "file, building, shot_date required"}, status=status.HTTP_400_BAD_REQUEST)

		item = ImageItem.objects.create(file=uploaded, building=building, shot_date=shot_date, notes=notes)

		vector = get_siglip().image_embed(item.file.path)

		payload = {
			"id": str(item.id),
			"building": item.building,
			"shot_date": str(item.shot_date),
			"shot_ymd": int(str(item.shot_date).replace('-', '')),
			"image_url": absolute_media_url(item.file.url),
			"notes": item.notes or "",
		}
		namespace = getattr(settings, 'PINECONE_NAMESPACE', '') or None
		get_vectors().upsert(str(item.id), vector, payload, namespace=namespace)

		return Response({"id": str(item.id), "image_url": payload["image_url"]}, status=status.HTTP_201_CREATED)


class ImageBatchIngestView(APIView):
	permission_classes = [permissions.AllowAny]

	def post(self, request):
		files = request.FILES.getlist('files')
		buildings = request.data.getlist('building')
		shot_dates = request.data.getlist('shot_date')
		notes_list = request.data.getlist('notes') if 'notes' in request.data else []

		if not files:
			return Response({"detail": "files required"}, status=400)
		if not (len(buildings) == len(files) and len(shot_dates) == len(files)):
			return Response({"detail": "building and shot_date arrays must match files count"}, status=400)
		if notes_list and len(notes_list) != len(files):
			notes_list = [''] * len(files)
		elif not notes_list:
			notes_list = [''] * len(files)

		items = []
		for idx, f in enumerate(files):
			item = ImageItem.objects.create(
				file=f,
				building=buildings[idx],
				shot_date=shot_dates[idx],
				notes=notes_list[idx] or ''
			)
			items.append(item)

		paths = [it.file.path for it in items]
		vectors = get_siglip().image_embed_batch(paths)

		upserts = []
		resp = []
		namespace = getattr(settings, 'PINECONE_NAMESPACE', '') or None
		for it, vec in zip(items, vectors):
			payload = {
				"id": str(it.id),
				"building": it.building,
				"shot_date": str(it.shot_date),
				"shot_ymd": int(str(it.shot_date).replace('-', '')),
				"image_url": absolute_media_url(it.file.url),
				"notes": it.notes or "",
			}
			upserts.append({"id": str(it.id), "values": vec, "metadata": payload})
			resp.append({"id": str(it.id), "image_url": payload["image_url"]})

		get_vectors().upsert_batch(upserts, namespace=namespace)
		return Response({"uploaded": resp, "namespace": namespace or ''}, status=201)


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
			embs = [get_siglip().text_embed(t) for t in terms]
			query_vec = np.mean(np.array(embs, dtype=float), axis=0).tolist()
		elif query_image_id:
			item = get_object_or_404(ImageItem, id=query_image_id)
			query_vec = get_siglip().image_embed(item.file.path)
		else:
			return Response({"detail": "provide q or query_image_id"}, status=status.HTTP_400_BAD_REQUEST)

		results = get_vectors().search(
			query_vec, top_k=top_k, building=building, date_from=date_from, date_to=date_to, namespace=namespace
		)
		results = rerank_with_metadata_boost(results, building)
		for r in results:
			if 'image_url' in r:
				r['image_url'] = absolute_media_url(r['image_url'])
		return Response({"results": results, "namespace": namespace or ''})
