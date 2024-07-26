FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential
RUN pip install "poetry==1.6.1"

COPY . .

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi


CMD [ "python", "main.py"]