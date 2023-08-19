FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y git

WORKDIR /tmp
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace