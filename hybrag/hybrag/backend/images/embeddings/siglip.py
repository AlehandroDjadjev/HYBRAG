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
        return None
    # Case 2: list of numbers -> treat as vector
    if isinstance(data, list) and data and all(isinstance(x, (int, float)) for x in data):
        return [float(x) for x in data]
    # Case 3: list of dicts (pick first dict with 'embedding')
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get('embedding'), list):
                vec = item.get('embedding')
                if vec and all(isinstance(x, (int, float)) for x in vec):
                    return [float(x) for x in vec]
    return None


class SiglipService:
	def __init__(self, model_name: str | None = None, device: str | None = None):
		# Ignore local model; use SageMaker endpoint per settings
		self.dim = int(getattr(settings, 'OS_EMB_DIM', 1536))
		endpoint = getattr(settings, 'SAGEMAKER_ENDPOINT_NAME', None)
		region = getattr(settings, 'SAGEMAKER_RUNTIME_REGION', None) or getattr(settings, 'AWS_REGION', None)
		if not endpoint:
			raise RuntimeError('SAGEMAKER_ENDPOINT_NAME not set')
		if not region:
			raise RuntimeError('SAGEMAKER_RUNTIME_REGION or AWS_REGION not set')
		self.endpoint = endpoint
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
		# If explicitly configured for async, use async path
		if self.use_async:
			bucket = getattr(settings, 'ASYNC_S3_INPUT_BUCKET', None) or getattr(settings, 'S3_BUCKET', None)
			prefix = getattr(settings, 'ASYNC_S3_INPUT_PREFIX', 'siglip2-async-inputs/')
			if not bucket:
				raise RuntimeError('Async invocation requested but no ASYNC_S3_INPUT_BUCKET or S3_BUCKET configured')
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
				obj = self.s3.get_object(Bucket=out_bucket, Key=out_key)
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
					obj = self.s3.get_object(Bucket=out_bucket, Key=out_key)
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
