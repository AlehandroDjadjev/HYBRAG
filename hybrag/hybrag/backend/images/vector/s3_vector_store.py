from __future__ import annotations
from typing import List, Optional, Dict, Any
import os
import json
import math
import boto3
from botocore.config import Config


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        va = float(a[i])
        vb = float(b[i])
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na == 0 or nb == 0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class S3VectorStore:
    def __init__(self, bucket: Optional[str] = None, index_name: str = 'images', dim: int = 1536, prefix: Optional[str] = None, region: Optional[str] = None):
        self.bucket = bucket or os.getenv('VECTOR_S3_BUCKET') or os.getenv('S3_BUCKET', '')
        if not self.bucket:
            raise RuntimeError('VECTOR_S3_BUCKET or S3_BUCKET must be set for S3VectorStore')
        self.index = index_name or os.getenv('VECTOR_S3_INDEX', 'images')
        self.dim = int(dim)
        self.prefix = (prefix or os.getenv('VECTOR_S3_PREFIX') or 'vectors/').strip('/')
        region_name = region or os.getenv('AWS_REGION') or 'eu-north-1'
        self.s3 = boto3.client('s3', config=Config(region_name=region_name, retries={"max_attempts": 3, "mode": "standard"}))

    def _key_for_id(self, point_id: str) -> str:
        return f"{self.prefix}/{self.index}/{point_id}.json"

    def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any], namespace: Optional[str] = None) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"Vector length {len(vector)} != dim {self.dim}")
        doc = {"id": point_id, **payload, "embedding": vector}
        body = json.dumps(doc).encode('utf-8')
        self.s3.put_object(Bucket=self.bucket, Key=self._key_for_id(point_id), Body=body, ContentType='application/json')

    def upsert_batch(self, items: List[Dict[str, Any]], namespace: Optional[str] = None) -> None:
        for it in items:
            vals = it.get('values') or it.get('vector')
            if vals is None:
                raise ValueError('Missing values for upsert item')
            if len(vals) != int(self.dim):
                raise ValueError(f"Vector length {len(vals)} != dim {self.dim}")
            doc = {"id": it.get('id'), **(it.get('metadata') or {}), "embedding": vals}
            body = json.dumps(doc).encode('utf-8')
            self.s3.put_object(Bucket=self.bucket, Key=self._key_for_id(str(it.get('id'))), Body=body, ContentType='application/json')

    def delete_ids(self, ids: List[str], namespace: Optional[str] = None) -> None:
        if not ids:
            return
        objs = [{"Key": self._key_for_id(i)} for i in ids]
        # Delete in chunks of 1000 (S3 limit)
        for i in range(0, len(objs), 1000):
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objs[i:i+1000]})

    def delete_all(self, namespace: Optional[str] = None) -> None:
        prefix = f"{self.prefix}/{self.index}/"
        token = None
        while True:
            res = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, ContinuationToken=token) if token else self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            contents = res.get('Contents') or []
            if contents:
                objs = [{"Key": it['Key']} for it in contents]
                for i in range(0, len(objs), 1000):
                    self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objs[i:i+1000]})
            if not res.get('IsTruncated'):
                break
            token = res.get('NextContinuationToken')

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        building: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        prefix = f"{self.prefix}/{self.index}/"
        token = None
        best: List[Dict[str, Any]] = []
        # Linear scan (sufficient for small-mid datasets); for large sets consider shard/manifest
        while True:
            res = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, ContinuationToken=token) if token else self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            contents = res.get('Contents') or []
            for obj in contents:
                b = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                try:
                    doc = json.loads(b['Body'].read())
                except Exception:
                    continue
                # Filters
                if building and doc.get('building') != building:
                    continue
                if date_from:
                    try:
                        ymd_from = int(date_from.replace('-', ''))
                        if int(doc.get('shot_ymd') or 0) < ymd_from:
                            continue
                    except Exception:
                        pass
                if date_to:
                    try:
                        ymd_to = int(date_to.replace('-', ''))
                        if int(doc.get('shot_ymd') or 0) > ymd_to:
                            continue
                    except Exception:
                        pass
                emb = doc.get('embedding')
                if not isinstance(emb, list) or len(emb) != self.dim:
                    continue
                score = _cosine(query_vector, emb)
                row = {"id": doc.get('id'), "score": score, **{k: v for k, v in doc.items() if k != 'embedding'}}
                best.append(row)
            if not res.get('IsTruncated'):
                break
            token = res.get('NextContinuationToken')
        best.sort(key=lambda x: float(x.get('score') or 0.0), reverse=True)
        return best[:max(1, int(top_k))]


