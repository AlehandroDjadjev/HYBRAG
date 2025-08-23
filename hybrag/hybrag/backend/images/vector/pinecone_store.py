from __future__ import annotations
from typing import List, Optional, Dict, Any
from pinecone import Pinecone
from pinecone.core.client.exceptions import NotFoundException

class VectorStore:
	def __init__(self, api_key: str, environment: Optional[str], index_name: str, dim: int, host: Optional[str] = None):
		self.pc = Pinecone(api_key=api_key)
		self.index_name = index_name
		self.dim = dim
		if host:
			self.index = self.pc.Index(host=host)
		else:
			try:
				self.index = self.pc.Index(self.index_name)
			except Exception as e:
				raise RuntimeError(
					"Pinecone index not found. Either set PINECONE_HOST for a serverless index host URL "
					"or pre-create the index in your project and ensure PINECONE_INDEX matches."
				) from e

	def _ensure_dim(self, vector: List[float]) -> None:
		if len(vector) != int(self.dim):
			raise ValueError(f"Vector length {len(vector)} does not match expected dim {self.dim}. Check model/index config.")

	def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any], namespace: Optional[str] = None) -> None:
		self._ensure_dim(vector)
		self.index.upsert(vectors=[{"id": point_id, "values": vector, "metadata": payload}], namespace=namespace)

	def upsert_batch(self, items: List[Dict[str, Any]], namespace: Optional[str] = None) -> None:
		for it in items:
			vals = it.get('values') or it.get('vector')
			if vals is None:
				raise ValueError('Missing values for upsert item')
			self._ensure_dim(vals)
		self.index.upsert(vectors=items, namespace=namespace)

	def delete_ids(self, ids: List[str], namespace: Optional[str] = None) -> None:
		if not ids:
			return
		self.index.delete(ids=ids, namespace=namespace)

	def delete_all(self, namespace: Optional[str] = None) -> None:
		try:
			self.index.delete(delete_all=True, namespace=namespace)
		except NotFoundException:
			return

	def search(
		self,
		query_vector: List[float],
		top_k: int = 10,
		building: Optional[str] = None,
		date_from: Optional[str] = None,
		date_to: Optional[str] = None,
		namespace: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		flt: Dict[str, Any] = {}
		if building:
			flt['building'] = {"$eq": building}
		date_range: Dict[str, int] = {}
		if date_from:
			try:
				date_range["$gte"] = int(date_from.replace('-', ''))
			except Exception:
				pass
		if date_to:
			try:
				date_range["$lte"] = int(date_to.replace('-', ''))
			except Exception:
				pass
		if date_range:
			flt['shot_ymd'] = date_range

		res = self.index.query(
			vector=query_vector,
			top_k=top_k,
			include_metadata=True,
			filter=flt or None,
			namespace=namespace,
		)
		matches = res.get('matches', []) if isinstance(res, dict) else getattr(res, 'matches', [])
		out: List[Dict[str, Any]] = []
		for m in matches:
			mid = m.get('id') if isinstance(m, dict) else getattr(m, 'id', None)
			score = m.get('score') if isinstance(m, dict) else getattr(m, 'score', None)
			metadata = m.get('metadata') if isinstance(m, dict) else getattr(m, 'metadata', {})
			row = {"id": str(mid), "score": float(score) if score is not None else 0.0}
			if isinstance(metadata, dict):
				row.update(metadata)
			out.append(row)
		return out
