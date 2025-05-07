# ---- Base Image ----
FROM python:3.10-slim

# ---- Set Working Directory ----
WORKDIR /app
    
# ---- Install System Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl && \
    rm -rf /var/lib/apt/lists/*
    
# ---- Copy Code ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
    
COPY ./src /app/src
    
# ---- Expose Streamlit Port ----
EXPOSE 8501
    
# ---- Start Command ----
CMD ["streamlit", "run", "src/langgraphagenticai/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    