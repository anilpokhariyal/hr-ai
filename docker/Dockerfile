FROM python:3.8.5
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY docker/gunicorn_conf.py /gunicorn_conf.py

WORKDIR /app/
COPY . /app
ENV PYTHONPATH=/app
#for flask migration
ENV FLASK_APP=hr.py

EXPOSE 5001

ENTRYPOINT ["/entrypoint.sh"]

