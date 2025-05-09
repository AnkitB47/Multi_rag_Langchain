# ---- Base Image ----
FROM python:3.10-slim

# ---- Set Working Directory ----
WORKDIR /app
    
# ---- Install System Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl && \
    rm -rf /var/lib/apt/lists/*
    
# ---- Copy Code & Install ----
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
    
COPY .streamlit /app/.streamlit  
COPY ./src /app/src
    
ENV PYTHONPATH=/app/src
ENV TMPDIR=/tmp
    
# ---- Expose Port & Start ----
EXPOSE 8501
CMD ["streamlit", "run", "src/langgraphagenticai/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    