from __future__ import annotations
from typing import List, Optional, Dict, Any
import os
import json
import math
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


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
        # Enable strict vector-bucket behavior when set
        self.vector_mode = os.getenv('VECTOR_S3_IS_VECTOR_BUCKET', '0') == '1'
        # Prefer explicit vector bucket region override, then fall back to general AWS region
        region_name = region or os.getenv('VECTOR_S3_REGION') or os.getenv('AWS_REGION') or 'eu-north-1'
        self.s3 = boto3.client('s3', config=Config(region_name=region_name, retries={"max_attempts": 3, "mode": "standard"}))
        # Detect actual bucket region and reinitialize client if needed
        bucket_region = None
        try:
            loc = self.s3.get_bucket_location(Bucket=self.bucket)
            # GetBucketLocation returns None for us-east-1
            if isinstance(loc, dict):
                bucket_region = loc.get('LocationConstraint')
            if not bucket_region:
                bucket_region = 'us-east-1'
            if bucket_region and bucket_region != region_name:
                self.s3 = boto3.client('s3', config=Config(region_name=bucket_region, retries={"max_attempts": 3, "mode": "standard"}))
        except ClientError as e:
            code = (e.response or {}).get('Error', {}).get('Code')
            if code == 'NoSuchBucket' and os.getenv('VECTOR_S3_CREATE', '0') == '1':
                params = {"Bucket": self.bucket}
                # For non-us-east-1, need LocationConstraint
                if region_name != 'us-east-1':
                    params["CreateBucketConfiguration"] = {"LocationConstraint": region_name}
                self.s3.create_bucket(**params)
        # Initialize S3 Vectors client in same region as bucket when possible
        try:
            self.s3vectors = boto3.client('s3vectors', region_name=bucket_region or region_name)
        except Exception as e:
            # In vector-bucket mode this is a fatal error (SDK too old or region mismatch)
            if self.vector_mode:
                raise RuntimeError(
                    f"s3vectors client unavailable. Ensure boto3/botocore support S3 Vectors and region is correct. Original: {e}"
                ) from e
            self.s3vectors = None

    def _key_for_id(self, point_id: str) -> str:
        return f"{self.prefix}/{self.index}/{point_id}.json"

    def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any], namespace: Optional[str] = None) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"Vector length {len(vector)} != dim {self.dim}")
        # Strict vector mode: do not fallback
        if self.vector_mode:
            self.s3vectors.put_vectors(
                vectorBucketName=self.bucket,
                indexName=self.index,
                vectors=[{
                    "key": str(point_id),
                    "data": {"float32": vector},
                    "metadata": dict(payload or {}),
                }],
            )
            return
        # Prefer S3 Vectors API; fallback to legacy S3 object if unavailable
        if self.s3vectors is not None:
            try:
                self.s3vectors.put_vectors(
                    vectorBucketName=self.bucket,
                    indexName=self.index,
                    vectors=[{
                        "key": str(point_id),
                        "data": {"float32": vector},
                        "metadata": dict(payload or {}),
                    }],
                )
                return
            except Exception:
                pass
        doc = {"id": point_id, **(payload or {}), "embedding": vector}
        body = json.dumps(doc).encode('utf-8')
        self.s3.put_object(Bucket=self.bucket, Key=self._key_for_id(point_id), Body=body, ContentType='application/json')

    def upsert_batch(self, items: List[Dict[str, Any]], namespace: Optional[str] = None) -> None:
        if not items:
            return
        # Strict vector mode: no fallback
        if self.vector_mode:
            batch: List[Dict[str, Any]] = []
            def flush() -> None:
                if not batch:
                    return
                self.s3vectors.put_vectors(
                    vectorBucketName=self.bucket,
                    indexName=self.index,
                    vectors=batch,
                )
                batch.clear()
            for it in items:
                vals = it.get('values') or it.get('vector')
                if vals is None:
                    raise ValueError('Missing values for upsert item')
                if len(vals) != int(self.dim):
                    raise ValueError(f"Vector length {len(vals)} != dim {self.dim}")
                batch.append({
                    "key": str(it.get('id')),
                    "data": {"float32": vals},
                    "metadata": dict((it.get('metadata') or {})),
                })
                if len(batch) >= 200:
                    flush()
            flush()
            return
        # Prefer S3 Vectors API with batching (legacy mode allows fallback)
        if self.s3vectors is not None:
            try:
                batch: List[Dict[str, Any]] = []
                def flush() -> None:
                    if not batch:
                        return
                    self.s3vectors.put_vectors(
                        vectorBucketName=self.bucket,
                        indexName=self.index,
                        vectors=batch,
                    )
                    batch.clear()
                for it in items:
                    vals = it.get('values') or it.get('vector')
                    if vals is None:
                        raise ValueError('Missing values for upsert item')
                    if len(vals) != int(self.dim):
                        raise ValueError(f"Vector length {len(vals)} != dim {self.dim}")
                    batch.append({
                        "key": str(it.get('id')),
                        "data": {"float32": vals},
                        "metadata": dict((it.get('metadata') or {})),
                    })
                    if len(batch) >= 200:
                        flush()
                flush()
                return
            except Exception:
                pass
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
        # Strict vector mode: no fallback
        if self.vector_mode:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i+500]
                self.s3vectors.delete_vectors(
                    vectorBucketName=self.bucket,
                    indexName=self.index,
                    keys=[str(k) for k in chunk],
                )
            return
        # Prefer S3 Vectors API (legacy mode allows fallback)
        if self.s3vectors is not None:
            try:
                for i in range(0, len(ids), 500):
                    chunk = ids[i:i+500]
                    self.s3vectors.delete_vectors(
                        vectorBucketName=self.bucket,
                        indexName=self.index,
                        keys=[str(k) for k in chunk],
                    )
                return
            except Exception:
                pass
        objs = [{"Key": self._key_for_id(i)} for i in ids]
        # Delete in chunks of 1000 (S3 limit)
        for i in range(0, len(objs), 1000):
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objs[i:i+1000]})

    def delete_all(self, namespace: Optional[str] = None) -> None:
        # Strict vector mode: list then delete with no fallback
        if self.vector_mode:
            next_token = None
            keys: List[str] = []
            while True:
                if next_token:
                    resp = self.s3vectors.list_vectors(
                        vectorBucketName=self.bucket,
                        indexName=self.index,
                        nextToken=next_token,
                    )
                else:
                    resp = self.s3vectors.list_vectors(
                        vectorBucketName=self.bucket,
                        indexName=self.index,
                    )
                for v in resp.get('vectors', []) or []:
                    k = v.get('key') if isinstance(v, dict) else None
                    if k is not None:
                        keys.append(str(k))
                next_token = resp.get('nextToken')
                if not next_token:
                    break
            if keys:
                self.delete_ids(keys, namespace=namespace)
            return
        # Prefer S3 Vectors API: list then delete (legacy mode allows fallback)
        if self.s3vectors is not None:
            try:
                next_token = None
                keys: List[str] = []
                while True:
                    if next_token:
                        resp = self.s3vectors.list_vectors(
                            vectorBucketName=self.bucket,
                            indexName=self.index,
                            nextToken=next_token,
                        )
                    else:
                        resp = self.s3vectors.list_vectors(
                            vectorBucketName=self.bucket,
                            indexName=self.index,
                        )
                    for v in resp.get('vectors', []) or []:
                        k = v.get('key') if isinstance(v, dict) else None
                        if k is not None:
                            keys.append(str(k))
                    next_token = resp.get('nextToken')
                    if not next_token:
                        break
                if keys:
                    self.delete_ids(keys, namespace=namespace)
                return
            except Exception:
                pass
        prefix = f"{self.prefix}/{self.index}/"
        token = None
        while True:
            try:
                res = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, ContinuationToken=token) if token else self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            except ClientError as e:
                code = (e.response or {}).get('Error', {}).get('Code')
                if code == 'NoSuchBucket':
                    return []
                raise
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
        # Guard on query vector length
        if len(query_vector) != self.dim:
            raise ValueError(f"Query vector length {len(query_vector)} != dim {self.dim}")
        # Strict vector mode: no fallback
        if self.vector_mode:
            filt: Dict[str, Any] = {}
            if building:
                filt["building"] = building
            resp = self.s3vectors.query_vectors(
                vectorBucketName=self.bucket,
                indexName=self.index,
                queryVector={"float32": query_vector},
                topK=int(max(1, top_k)),
                filter=filt or None,
                returnDistance=True,
                returnMetadata=True,
            )
            vectors = resp.get('vectors', []) or []
            results: List[Dict[str, Any]] = []
            for v in vectors:
                key = v.get('key') if isinstance(v, dict) else None
                meta = v.get('metadata') if isinstance(v, dict) else {}
                dist = v.get('distance') if isinstance(v, dict) else None
                row: Dict[str, Any] = {"id": str(key) if key is not None else None}
                if isinstance(dist, (int, float)):
                    try:
                        row["score"] = 1.0 - float(dist)
                    except Exception:
                        row["score"] = float(0.0)
                if isinstance(meta, dict):
                    row.update(meta)
                results.append(row)
            out: List[Dict[str, Any]] = []
            for r in results:
                if date_from:
                    try:
                        ymd_from = int(str(date_from).replace('-', ''))
                        if int(r.get('shot_ymd') or 0) < ymd_from:
                            continue
                    except Exception:
                        pass
                if date_to:
                    try:
                        ymd_to = int(str(date_to).replace('-', ''))
                        if int(r.get('shot_ymd') or 0) > ymd_to:
                            continue
                    except Exception:
                        pass
                out.append(r)
            out.sort(key=lambda x: float(x.get('score') or 0.0), reverse=True)
            return out[:max(1, int(top_k))]
        # Prefer S3 Vectors API (legacy mode allows fallback)
        if self.s3vectors is not None:
            try:
                filt: Dict[str, Any] = {}
                if building:
                    filt["building"] = building
                resp = self.s3vectors.query_vectors(
                    vectorBucketName=self.bucket,
                    indexName=self.index,
                    queryVector={"float32": query_vector},
                    topK=int(max(1, top_k)),
                    filter=filt or None,
                    returnDistance=True,
                    returnMetadata=True,
                )
                vectors = resp.get('vectors', []) or []
                results: List[Dict[str, Any]] = []
                for v in vectors:
                    key = v.get('key') if isinstance(v, dict) else None
                    meta = v.get('metadata') if isinstance(v, dict) else {}
                    dist = v.get('distance') if isinstance(v, dict) else None
                    row: Dict[str, Any] = {"id": str(key) if key is not None else None}
                    if isinstance(dist, (int, float)):
                        try:
                            row["score"] = 1.0 - float(dist)
                        except Exception:
                            row["score"] = float(0.0)
                    if isinstance(meta, dict):
                        row.update(meta)
                    results.append(row)
                out: List[Dict[str, Any]] = []
                for r in results:
                    if date_from:
                        try:
                            ymd_from = int(str(date_from).replace('-', ''))
                            if int(r.get('shot_ymd') or 0) < ymd_from:
                                continue
                        except Exception:
                            pass
                    if date_to:
                        try:
                            ymd_to = int(str(date_to).replace('-', ''))
                            if int(r.get('shot_ymd') or 0) > ymd_to:
                                continue
                        except Exception:
                            pass
                    out.append(r)
                out.sort(key=lambda x: float(x.get('score') or 0.0), reverse=True)
                return out[:max(1, int(top_k))]
            except Exception:
                pass
        # Fallback: legacy JSON scan in S3
        prefix = f"{self.prefix}/{self.index}/"
        token = None
        best: List[Dict[str, Any]] = []
        while True:
            res = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, ContinuationToken=token) if token else self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            contents = res.get('Contents') or []
            for obj in contents:
                b = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                try:
                    doc = json.loads(b['Body'].read())
                except Exception:
                    continue
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


