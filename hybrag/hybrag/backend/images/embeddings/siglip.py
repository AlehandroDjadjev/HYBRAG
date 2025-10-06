from __future__ import annotations
from typing import List
from django.conf import settings
import json
import boto3
from botocore.config import Config


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

	def _invoke(self, payload: dict) -> list[float]:
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

	def image_embed(self, file_path: str) -> list[float]:
		# This code used to load image locally. Now assume image is accessible via URL.
		# For backward compatibility, we keep signature but raise to avoid silently reading local files.
		raise RuntimeError('Local image embedding disabled. Use presigned S3 GET and image_url via the ingestion flow.')

	def image_embed_batch(self, file_paths: List[str]) -> List[List[float]]:
		raise RuntimeError('Local batch embedding disabled. Use S3 presigned URLs and invoke per image or batch upstream.')

	def text_embed(self, text: str) -> list[float]:
		return self._invoke({"text": text, "normalize": True})
