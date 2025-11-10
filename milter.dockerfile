FROM python:3.13-slim AS pip

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests --assume-yes libmilter-dev build-essential git && rm -rf /var/*/apt/
COPY ./requirements.txt ./src/requirements.txt
RUN pip install --no-cache-dir -Ur ./src/requirements.txt

COPY ./spamscouter/ ./src/spamscouter/
COPY ./pyproject.toml ./src/pyproject.toml
RUN pip install --no-cache-dir ./src/

FROM python:3.13-slim

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests --assume-yes libmilter-dev pandoc && rm -rf /var/*/apt/
COPY --from=pip /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/

EXPOSE 3639
ENTRYPOINT [ "python3", "-um", "spamscouter.milter", "/var/lib/spamscouter/", "--address", "3639" ]
