FROM python:3.10.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD sh -c "mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port ${PORT:-10000}"