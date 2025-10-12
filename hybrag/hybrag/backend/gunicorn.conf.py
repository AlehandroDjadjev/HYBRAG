import os

# Env-driven knobs
proc_name = os.getenv("NAME", "hybrag")
workers = int(os.getenv("NUM_WORKERS", "3"))
timeout = int(os.getenv("TIMEOUT", "120"))
bind = os.getenv("BIND", "0.0.0.0:8000")
loglevel = os.getenv("LOG_LEVEL", "debug")
pidfile = os.getenv("PIDFILE", "/tmp/gunicorn.pid")

# Django WSGI module
wsgi_app = f"{os.getenv('DJANGO_WSGI_MODULE', 'server.wsgi')}:application"

# Logs to stdout/stderr for App Runner
accesslog = "-"
errorlog = "-"




