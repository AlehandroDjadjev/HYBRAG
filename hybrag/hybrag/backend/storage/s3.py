from __future__ import annotations
from typing import Dict
import boto3
from botocore.config import Config
from django.conf import settings

_region = getattr(settings, 'AWS_REGION', 'eu-north-1')
_bucket = getattr(settings, 'S3_BUCKET', '')
_expire = int(getattr(settings, 'S3_PRESIGN_EXPIRE', 3600))

_s3 = boto3.client(
    's3',
    config=Config(
        region_name=_region,
        signature_version='s3v4',
        s3={'addressing_style': 'virtual'},
        retries={"max_attempts": 3, "mode": "standard"},
    ),
)


def presign_put(key: str, content_type: str = 'image/jpeg') -> Dict:
	if not _bucket:
		raise ValueError('S3_BUCKET not configured')
	url = _s3.generate_presigned_url(
		'put_object',
		Params={'Bucket': _bucket, 'Key': key, 'ContentType': content_type},
		ExpiresIn=_expire,
	)
	return {'url': url, 'bucket': _bucket, 'key': key, 'expires_in': _expire}


def presign_get(key: str) -> str:
	if not _bucket:
		raise ValueError('S3_BUCKET not configured')
	return _s3.generate_presigned_url(
		'get_object',
		Params={'Bucket': _bucket, 'Key': key},
        ExpiresIn=min(_expire, 900),
	)

