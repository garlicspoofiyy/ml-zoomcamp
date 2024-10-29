# Codes ran in the terminal:

# docker run -it --rm python:3.12.7-slim
# docker run -it --rm --entrypoint=bash python:3.12.7-slim
# docker build -t zoomcamp-test .
# docker builder prune
# docker run -it --rm -p 9696:9696 zoomcamp-test

# Why do we need docker
# Running a python image with docker
# Dockerfile
# Building a docker image
# Running a docker image

FROM python:3.12.7-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
