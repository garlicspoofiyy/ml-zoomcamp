FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv
WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["homework.py", "model1.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "homework:app"]