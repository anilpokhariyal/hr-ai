#!/bin/sh

DEFAULT_GUNICORN_CONF=/gunicorn_conf.py
export GUNICORN_CONF=${GUNICORN_CONF:-$DEFAULT_GUNICORN_CONF}
export  prometheus_multiproc_dir="/tmp"

exec gunicorn hr:app -b 0.0.0.0:5001 -c "$GUNICORN_CONF" --reload --timeout 300 --workers $GUNICORN_WORKERS --worker-class gevent --worker-connections 1000
