from __future__ import annotations
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, Range

class VectorStore:
	def __init__(self, host: str, port: int, collection: str, dim: int, url: Optional[str] = None, api_key: Optional[str] = None):
		if url:
			self.client = QdrantClient(url=url, api_key=api_key)
		else:
			self.client = QdrantClient(host=host, port=port, api_key=api_key)
		self.collection = collection
		self.dim = dim
		self._ensure_collection()

	def _ensure_collection(self) -> None:
		collections = {c.name for c in self.client.get_collections().collections}
		if self.collection not in collections:
			self.client.create_collection(
				collection_name=self.collection,
				vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
			)

	def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
		self.client.upsert(
			collection_name=self.collection,
			points=[{"id": point_id, "vector": vector, "payload": payload}],
		)

	def search(
		self,
		query_vector: List[float],
		top_k: int = 10,
		building: Optional[str] = None,
		date_from: Optional[str] = None,
		date_to: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		conditions: List[Any] = []
		if building:
			conditions.append(FieldCondition(key="building", match=MatchValue(value=building)))
		if date_from:
			try:
				conditions.append(FieldCondition(key="shot_ymd", range=Range(gte=int(date_from.replace('-', '')))))
			except Exception:
				pass
		if date_to:
			try:
				conditions.append(FieldCondition(key="shot_ymd", range=Range(lte=int(date_to.replace('-', '')))))
			except Exception:
				pass

		flt = Filter(must=conditions) if conditions else None
		res = self.client.search(
			collection_name=self.collection,
			query_vector=query_vector,
			limit=top_k,
			query_filter=flt,
		)
		return [
			{"id": str(p.id), "score": float(p.score), **(p.payload or {})}
			for p in res
		]
