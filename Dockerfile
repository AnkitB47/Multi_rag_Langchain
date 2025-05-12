# ---- Base Image ----
FROM python:3.10-slim

# ---- Working Directory ----
WORKDIR /app
    
# ---- System Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl && \
    rm -rf /var/lib/apt/lists/*
    
# ---- Python Dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    
# ---- App Code ----
COPY ./src /app/src
COPY .streamlit /app/.streamlit
    
# ---- Environment ----
ENV PYTHONPATH=/app/src
ENV TMPDIR=/tmp
    
# ---- Expose Streamlit Port ----
EXPOSE 8501
    
# ---- Create /tmp and set permissions ----
RUN mkdir -p /tmp && chmod 777 /tmp
    
# ---- Launch ----
CMD ["streamlit", "run", "src/langgraphagenticai/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
