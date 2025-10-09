import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-change-me')
DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
ALLOWED_HOSTS = [".awsapprunner.com"]

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'images',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'server.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'server.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}
MEDIA_ROOT = os.getenv('MEDIA_ROOT', BASE_DIR / 'media')
MEDIA_URL = os.getenv('MEDIA_URL', '/media/')
# Public base URL where this backend is reachable (for building absolute file URLs)
BACKEND_BASE_URL = os.getenv('BACKEND_BASE_URL', 'http://127.0.0.1:8000')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CORS_ALLOWED_ORIGINS = [o for o in os.getenv('CORS_ALLOW_ORIGINS', '').split(',') if o]
CORS_ALLOW_ALL_ORIGINS = not CORS_ALLOWED_ORIGINS

REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': ['rest_framework.renderers.JSONRenderer'],
}

# Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'construction-images')
PINECONE_HOST = os.getenv('PINECONE_HOST')
PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE', '')
PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')

# Embeddings
SIGLIP_MODEL_NAME = os.getenv('SIGLIP_MODEL_NAME', 'google/siglip2-giant-opt-patch16-384')
DEVICE = os.getenv('DEVICE', 'auto')

# SageMaker
SAGEMAKER_ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME')
SAGEMAKER_RUNTIME_REGION = os.getenv('SAGEMAKER_RUNTIME_REGION') or os.getenv('AWS_REGION')
EMBEDDINGS_PROVIDER = os.getenv('SIGLIP_PROVIDER', 'local')  # 'local' or 'sagemaker'
AWS_REGION = os.getenv('AWS_REGION', SAGEMAKER_RUNTIME_REGION or 'eu-north-1')
SAGEMAKER_ASYNC = os.getenv('SAGEMAKER_ASYNC', '1') == '1'
# Async polling timeout (seconds)
SAGEMAKER_ASYNC_TIMEOUT = int(os.getenv('SAGEMAKER_ASYNC_TIMEOUT', '150'))

# S3 (presigned URLs)
S3_BUCKET = os.getenv('S3_BUCKET', '')
S3_PRESIGN_EXPIRE = int(os.getenv('S3_PRESIGN_EXPIRE', '3600'))
ASYNC_S3_INPUT_BUCKET = os.getenv('ASYNC_S3_INPUT_BUCKET', S3_BUCKET)
ASYNC_S3_INPUT_PREFIX = os.getenv('ASYNC_S3_INPUT_PREFIX', 'siglip2-async-inputs/')

# OpenSearch
OS_HOST = os.getenv('OS_HOST', '')  # e.g., https://search-... or https://<vpce>...
OS_USERNAME = os.getenv('OS_USERNAME', '')
OS_PASSWORD = os.getenv('OS_PASSWORD', '')
OS_USE_IAM = os.getenv('OS_USE_IAM', '1') == '1'
OS_INDEX = os.getenv('OS_INDEX', 'media-embeddings')
OS_EMB_DIM = int(os.getenv('OS_EMB_DIM', '1536'))
OS_K = int(os.getenv('OS_K', '10'))