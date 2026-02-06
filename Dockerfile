
FROM python:3.12-slim

WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
# We use pip since we are inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ src/
COPY models/ models/
COPY app.py .
COPY ui.py .

# Environment variables
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000
EXPOSE 7860

# Default command: run FastAPI
# Users can override this to run Gradio: python ui.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
