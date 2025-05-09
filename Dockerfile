FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY ./src /app/src
COPY .streamlit /app/.streamlit

ENV PYTHONPATH=/app/src
ENV TMPDIR=/tmp

RUN mkdir -p /tmp && chmod 777 /tmp

EXPOSE 8501

CMD ["streamlit", "run", "src/langgraphagenticai/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
