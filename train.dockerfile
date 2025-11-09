FROM python:3.13-slim

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests --assume-yes libmilter-dev build-essential pandoc git && rm -rf /var/*/apt/
COPY ./requirements.txt ./src/requirements.txt
RUN pip install --no-cache-dir -Ur ./src/requirements.txt

COPY ./spamscouter/ ./src/spamscouter/
COPY ./pyproject.toml ./src/pyproject.toml
RUN pip install --no-cache-dir ./src/

ENTRYPOINT [ "python3", "/src/trainer.py" ]
