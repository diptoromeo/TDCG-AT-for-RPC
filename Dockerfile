FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY main.py .
RUN mkdir -p data/raw data/processed data/external
EXPOSE 8000
ENV DATA_DIR=/app/data/raw
ENV PROC_DIR=/app/data/processed
ENV EXT_DIR=/app/data/external
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
