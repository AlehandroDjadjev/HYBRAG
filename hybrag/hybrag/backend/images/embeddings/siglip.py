from __future__ import annotations
from typing import List, Optional
from django.conf import settings
import json
import boto3
from botocore.config import Config
import time
import uuid


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith('s3://'):
        raise ValueError(f'Not an s3 uri: {uri}')
    without = uri[len('s3://') :]
    parts = without.split('/', 1)
    if len(parts) != 2:
        raise ValueError(f'Invalid s3 uri: {uri}')
    return parts[0], parts[1]


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
		self.use_async = bool(getattr(settings, 'SAGEMAKER_ASYNC', False))
		self.async_input_bucket: Optional[str] = getattr(settings, 'ASYNC_S3_INPUT_BUCKET', None)
		self.async_input_prefix: str = getattr(settings, 'ASYNC_S3_INPUT_PREFIX', 'siglip2-async-inputs/')

	def _invoke(self, payload: dict) -> list[float]:
		if not self.use_async:
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

		# Async path: upload payload to S3, invoke async, poll output
		if not self.async_input_bucket:
			raise RuntimeError('ASYNC_S3_INPUT_BUCKET must be set for async invocation')
		key = f"{self.async_input_prefix.rstrip('/')}/{uuid.uuid4()}.json"
		body_bytes = json.dumps(payload).encode('utf-8')
		self.s3.put_object(Bucket=self.async_input_bucket, Key=key, Body=body_bytes, ContentType='application/json')
		in_loc = f"s3://{self.async_input_bucket}/{key}"
		resp = self.client.invoke_endpoint_async(
			EndpointName=self.endpoint,
			InputLocation=in_loc,
			ContentType='application/json',
		)
		out_loc = resp.get('OutputLocation')
		if not out_loc:
			raise RuntimeError(f'Async invoke returned no OutputLocation: {resp}')
		out_bucket, out_key = _parse_s3_uri(out_loc)
		# Poll for result
		deadline = time.time() + 120
		last_err: Optional[str] = None
		while time.time() < deadline:
			try:
				obj = self.s3.get_object(Bucket=out_bucket, Key=out_key)
				data = json.loads(obj['Body'].read())
				vec = data.get('embedding')
				if not isinstance(vec, list):
					last_err = f'Bad async response: {data}'
				else:
					return vec
			except self.s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
				pass
			except Exception as e:
				last_err = str(e)
			time.sleep(2)
		raise RuntimeError(last_err or 'Timed out waiting for async output')

	def image_embed(self, file_path: str) -> list[float]:
		# This code used to load image locally. Now assume image is accessible via URL.
		# For backward compatibility, we keep signature but raise to avoid silently reading local files.
		raise RuntimeError('Local image embedding disabled. Use presigned S3 GET and image_url via the ingestion flow.')

	def image_embed_batch(self, file_paths: List[str]) -> List[List[float]]:
		raise RuntimeError('Local batch embedding disabled. Use S3 presigned URLs and invoke per image or batch upstream.')

	def text_embed(self, text: str) -> list[float]:
		return self._invoke({"text": text, "normalize": True})
