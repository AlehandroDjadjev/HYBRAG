from __future__ import annotations
from typing import List
from django.conf import settings
import json
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import time
import uuid


def _parse_s3_uri(uri: str) -> tuple[str, str]:
	if not uri.startswith('s3://'):
		raise ValueError(f'Not an s3 uri: {uri}')
	without = uri[len('s3://'):]
	parts = without.split('/', 1)
	if len(parts) != 2:
		raise ValueError(f'Invalid s3 uri: {uri}')
	return parts[0], parts[1]


def _extract_embedding_from_data(data) -> List[float] | None:
    # Case 1: dict with 'embedding'
    if isinstance(data, dict):
        vec = data.get('embedding')
        if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
            return [float(x) for x in vec]
        # Sometimes body is nested JSON string
        body = data.get('body') or data.get('Body')
        if isinstance(body, str):
            try:
                inner = json.loads(body)
                return _extract_embedding_from_data(inner)
            except Exception:
                return None
        return None
    # Case 2: list of numbers -> treat as vector
    if isinstance(data, list) and data and all(isinstance(x, (int, float)) for x in data):
        return [float(x) for x in data]
    # Case 3: list of dicts (pick first dict with 'embedding' or nested body)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                vec = item.get('embedding')
                if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                    return [float(x) for x in vec]
                body = item.get('body') or item.get('Body')
                if isinstance(body, str):
                    try:
                        inner = json.loads(body)
                        maybe = _extract_embedding_from_data(inner)
                        if maybe is not None:
                            return maybe
                    except Exception:
                        pass
            elif isinstance(item, str):
                try:
                    inner = json.loads(item)
                    maybe = _extract_embedding_from_data(inner)
                    if maybe is not None:
                        return maybe
                except Exception:
                    pass
    return None


class SiglipService:
	def __init__(self, model_name: str | None = None, device: str | None = None):
		# Ignore local model; use SageMaker endpoint per settings
		self.dim = int(getattr(settings, 'OS_EMB_DIM', 1536))
		# Support separate endpoints for text (realtime) and image (async)
		text_ep = getattr(settings, 'SAGEMAKER_TEXT_ENDPOINT_NAME', None) or getattr(settings, 'SAGEMAKER_ENDPOINT_NAME', None)
		image_ep = getattr(settings, 'SAGEMAKER_IMAGE_ENDPOINT_NAME', None) or getattr(settings, 'SAGEMAKER_ENDPOINT_NAME', None)
		region = getattr(settings, 'SAGEMAKER_RUNTIME_REGION', None) or getattr(settings, 'AWS_REGION', None)
		if not text_ep and not image_ep:
			raise RuntimeError('SAGEMAKER_TEXT_ENDPOINT_NAME or SAGEMAKER_IMAGE_ENDPOINT_NAME (or SAGEMAKER_ENDPOINT_NAME) must be set')
		if not region:
			raise RuntimeError('SAGEMAKER_RUNTIME_REGION or AWS_REGION not set')
		self.text_endpoint = text_ep
		self.image_endpoint = image_ep
		self.client = boto3.client(
			'sagemaker-runtime',
			config=Config(
				region_name=region,
				retries={"max_attempts": 3, "mode": "standard"},
				read_timeout=12,
				connect_timeout=3,
			)
		)
		self.s3 = boto3.client('s3', config=Config(region_name=getattr(settings, 'AWS_REGION', region)))
		self.use_async = True

	def _invoke(self, payload: dict) -> list[float]:
		# Route by payload: text -> realtime endpoint, image -> async endpoint
		is_image = any(k in payload for k in ("image_url", "image_urls"))
		if is_image:
			endpoint_name = self.image_endpoint
			if not endpoint_name:
				raise RuntimeError('SAGEMAKER_IMAGE_ENDPOINT_NAME not configured')
			# Async invocation flow for image embeddings
			bucket = getattr(settings, 'ASYNC_S3_INPUT_BUCKET', None) or getattr(settings, 'S3_BUCKET', None)
			prefix = getattr(settings, 'ASYNC_S3_INPUT_PREFIX', 'siglip2-async-inputs/')
			if not bucket:
				raise RuntimeError('Async invocation requested but no ASYNC_S3_INPUT_BUCKET or S3_BUCKET configured')
			key = f"{prefix.rstrip('/')}/{uuid.uuid4()}.json"
			self.s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode('utf-8'), ContentType='application/json')
			in_loc = f"s3://{bucket}/{key}"
			resp_async = self.client.invoke_endpoint_async(
				EndpointName=endpoint_name,
				InputLocation=in_loc,
				ContentType='application/json',
			)
			out_loc = resp_async.get('OutputLocation')
			if out_loc:
				out_bucket, out_key = _parse_s3_uri(out_loc)
			else:
				# Some async setups rely on known output prefix; derive path from request key
				out_bucket = getattr(settings, 'ASYNC_S3_OUTPUT_BUCKET', None) or bucket
				out_prefix = getattr(settings, 'ASYNC_S3_OUTPUT_PREFIX', 'siglip2-async-outputs/')
				# Mirror filename, allow service to append suffixes; we will list and pick the newest
				base_name = key.rsplit('/', 1)[-1].rsplit('.', 1)[0]
				candidate_prefix = f"{out_prefix.rstrip('/')}/{base_name}"
				# We'll list with prefix and pick first object when available
			deadline = time.time() + int(getattr(settings, 'SAGEMAKER_ASYNC_TIMEOUT', 150))
			last_err = None
			while time.time() < deadline:
				try:
					if out_loc:
						obj = self.s3.get_object(Bucket=out_bucket, Key=out_key)
						data = json.loads(obj['Body'].read())
					else:
						# List objects under derived prefix and read the first available
						lst = self.s3.list_objects_v2(Bucket=out_bucket, Prefix=candidate_prefix)
						contents = lst.get('Contents') or []
						if not contents:
							raise self.s3.exceptions.NoSuchKey({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')  # trigger retry
						best = sorted(contents, key=lambda x: x.get('LastModified') or 0, reverse=True)[0]
						obj = self.s3.get_object(Bucket=out_bucket, Key=best['Key'])
						data = json.loads(obj['Body'].read())
					vec = _extract_embedding_from_data(data)
					if vec is not None:
						return vec
					last_err = f'Bad async response shape: {type(data).__name__}'
				except self.s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
					pass
				except Exception as pe:
					last_err = str(pe)
				time.sleep(2)
			raise RuntimeError(last_err or 'Timed out waiting for async output')
		# Realtime invocation for text embeddings
		endpoint_name = self.text_endpoint
		if not endpoint_name:
			raise RuntimeError('SAGEMAKER_TEXT_ENDPOINT_NAME not configured')
		resp = self.client.invoke_endpoint(
			EndpointName=endpoint_name,
			ContentType='application/json',
			Accept='application/json',
			Body=json.dumps(payload).encode('utf-8'),
		)
		body = json.loads(resp['Body'].read())
		vec = _extract_embedding_from_data(body) or body.get('embedding')
		if not isinstance(vec, list):
			raise RuntimeError(f'Bad response from SageMaker endpoint: {body}')
		return vec

		# Try real-time first, fallback to async if the endpoint rejects it
		try:
			resp = self.client.invoke_endpoint(
				EndpointName=self.endpoint,
				ContentType='application/json',
				Accept='application/json',
				Body=json.dumps(payload).encode('utf-8'),
			)
			body = json.loads(resp['Body'].read())
			vec = body.get('embedding')
			if not isinstance(vec, list):
				raise RuntimeError(f'Bad response from SageMaker endpoint: {body}')
			return vec
		except ClientError as e:
			msg = str(e)
			if 'does not support this inference type' in msg:
				# Fallback to async
				bucket = getattr(settings, 'ASYNC_S3_INPUT_BUCKET', None) or getattr(settings, 'S3_BUCKET', None)
				prefix = getattr(settings, 'ASYNC_S3_INPUT_PREFIX', 'siglip2-async-inputs/')
				if not bucket:
					raise RuntimeError('Async fallback requested but no ASYNC_S3_INPUT_BUCKET or S3_BUCKET configured')
				key = f"{prefix.rstrip('/')}/{uuid.uuid4()}.json"
				self.s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode('utf-8'), ContentType='application/json')
				in_loc = f"s3://{bucket}/{key}"
				resp_async = self.client.invoke_endpoint_async(
					EndpointName=self.endpoint,
					InputLocation=in_loc,
					ContentType='application/json',
				)
				out_loc = resp_async.get('OutputLocation')
				if not out_loc:
					raise RuntimeError(f'Async invoke returned no OutputLocation: {resp_async}')
				out_bucket, out_key = _parse_s3_uri(out_loc)
				deadline = time.time() + int(getattr(settings, 'SAGEMAKER_ASYNC_TIMEOUT', 150))
				last_err = None
				while time.time() < deadline:
					try:
						if out_loc:
							obj = self.s3.get_object(Bucket=out_bucket, Key=out_key)
							data = json.loads(obj['Body'].read())
						else:
							lst = self.s3.list_objects_v2(Bucket=out_bucket, Prefix=candidate_prefix)
							contents = lst.get('Contents') or []
							if not contents:
								raise self.s3.exceptions.NoSuchKey({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
							best = sorted(contents, key=lambda x: x.get('LastModified') or 0, reverse=True)[0]
							obj = self.s3.get_object(Bucket=out_bucket, Key=best['Key'])
							data = json.loads(obj['Body'].read())
						vec = _extract_embedding_from_data(data)
						if vec is not None:
							return vec
						last_err = f'Bad async response shape: {type(data).__name__}'
					except self.s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
						pass
					except Exception as pe:
						last_err = str(pe)
					time.sleep(2)
				raise RuntimeError(last_err or 'Timed out waiting for async output')
			raise

	def image_embed(self, file_path: str) -> list[float]:
		# This code used to load image locally. Now assume image is accessible via URL.
		# For backward compatibility, we keep signature but raise to avoid silently reading local files.
		raise RuntimeError('Local image embedding disabled. Use presigned S3 GET and image_url via the ingestion flow.')

	def image_embed_batch(self, file_paths: List[str]) -> List[List[float]]:
		raise RuntimeError('Local batch embedding disabled. Use S3 presigned URLs and invoke per image or batch upstream.')

	def text_embed(self, text: str) -> list[float]:
		return self._invoke({"text": text, "normalize": True})
