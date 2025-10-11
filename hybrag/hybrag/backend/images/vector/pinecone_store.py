from __future__ import annotations
from typing import List, Optional, Dict, Any
import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

class VectorStore:
	def __init__(self, api_key: str = '', environment: Optional[str] = None, index_name: str = 'media-embeddings', dim: int = 1536, host: Optional[str] = None):
		# Repurpose to OpenSearch. Ignore Pinecone args except index_name and dim.
		self.index_name = index_name
		self.dim = dim
		os_host = os.getenv('OS_HOST') or host
		if not os_host:
			raise RuntimeError('OS_HOST must be set to use OpenSearch')
		use_iam = (os.getenv('OS_USE_IAM', '1') == '1')
		region = os.getenv('OS_REGION') or os.getenv('AWS_REGION', 'us-east-1')
		# Derive region from host if not provided (e.g., https://...aos.us-east-1.on.aws)
		if not os.getenv('OS_REGION') and isinstance(os_host, str) and '.on.aws' in os_host:
			try:
				parts = os_host.split('.')
				for i, p in enumerate(parts):
					if p in ("aos", "es", "amazonaws", "on") and i > 0:
						cand = parts[i-1]
						if cand and '-' in cand:
							region = cand
							break
			except Exception:
				pass
		if use_iam:
			session = boto3.Session()
			creds = session.get_credentials()
			if creds is None:
				raise RuntimeError('No AWS credentials for OpenSearch IAM auth')
			awsauth = AWS4Auth(creds.access_key, creds.secret_key, region, 'es', session_token=creds.token)
			self.client = OpenSearch(
				hosts=[os_host],
				http_auth=awsauth,
				use_ssl=True,
				verify_certs=True,
				connection_class=RequestsHttpConnection,
				timeout=10,
				max_retries=3,
				retry_on_timeout=True,
			)
		else:
			user = os.getenv('OS_USERNAME', '')
			pwd = os.getenv('OS_PASSWORD', '')
			self.client = OpenSearch(
				hosts=[os_host],
				http_auth=(user, pwd),
				use_ssl=True,
				verify_certs=True,
				connection_class=RequestsHttpConnection,
				timeout=10,
				max_retries=3,
				retry_on_timeout=True,
			)
		self._ensure_index()

	def _ensure_index(self) -> None:
		if self.client.indices.exists(index=self.index_name):
			return
		body = {
			"settings": {"index": {"knn": True}},
			"mappings": {
				"properties": {
					"id": {"type": "keyword"},
					"building": {"type": "keyword"},
					"shot_date": {"type": "keyword"},
					"shot_ymd": {"type": "integer"},
					"image_url": {"type": "keyword"},
					"notes": {"type": "text"},
					"embedding": {
						"type": "knn_vector",
						"dimension": self.dim,
						"method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}
					}
				}
			}
		}
		self.client.indices.create(index=self.index_name, body=body)

	def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any], namespace: Optional[str] = None) -> None:
		# namespace ignored in OS simple setup
		doc = {"id": point_id, **payload, "embedding": vector}
		self.client.index(index=self.index_name, id=point_id, body=doc, refresh="wait_for")

	def upsert_batch(self, items: List[Dict[str, Any]], namespace: Optional[str] = None) -> None:
		for it in items:
			vals = it.get('values') or it.get('vector')
			if vals is None:
				raise ValueError('Missing values for upsert item')
			if len(vals) != int(self.dim):
				raise ValueError(f"Vector length {len(vals)} != dim {self.dim}")
			doc = {"id": it.get('id'), **(it.get('metadata') or {}), "embedding": vals}
			self.client.index(index=self.index_name, id=it.get('id'), body=doc, refresh=False)
		self.client.indices.refresh(index=self.index_name)

	def delete_ids(self, ids: List[str], namespace: Optional[str] = None) -> None:
		if not ids:
			return
		for id in ids:
			self.client.delete(index=self.index_name, id=id)
		self.client.indices.refresh(index=self.index_name)

	def delete_all(self, namespace: Optional[str] = None) -> None:
		self.client.indices.delete(index=self.index_name)

	def search(
		self,
		query_vector: List[float],
		top_k: int = 10,
		building: Optional[str] = None,
		date_from: Optional[str] = None,
		date_to: Optional[str] = None,
		namespace: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		filters: Dict[str, Any] = {}
		must: List[Any] = []
		if building:
			must.append({"term": {"building": building}})
		if date_from or date_to:
			rng: Dict[str, Any] = {}
			if date_from:
				try:
					rng["gte"] = int(date_from.replace('-', ''))
				except Exception:
					pass
			if date_to:
				try:
					rng["lte"] = int(date_to.replace('-', ''))
				except Exception:
					pass
			if rng:
				must.append({"range": {"shot_ymd": rng}})
		if must:
			filters = {"bool": {"must": must}}

		query: Dict[str, Any] = {
			"size": top_k,
			"query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
			"knn": {"embedding": {"vector": query_vector, "k": top_k}},
		}
		res = self.client.search(index=self.index_name, body=query)
		hits = res.get('hits', {}).get('hits', [])
		out: List[Dict[str, Any]] = []
		for h in hits:
			row = {"id": h.get('_id'), "score": float(h.get('_score') or 0.0)}
			src = h.get('_source') or {}
			row.update(src)
			out.append(row)
		return out
