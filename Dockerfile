FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for opencv and tensorflow (minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libglib2.0-0 libsm6 libxext6 libxrender1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY Requirements.txt /app/Requirements.txt
RUN pip install --upgrade pip setuptools wheel && pip install -r /app/Requirements.txt

# copy app files
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
