FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image small (the free hosting tier has no GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY app.py .
COPY static ./static
COPY model ./model

ENV PORT=10000
EXPOSE 10000

# One worker: Grad-CAM uses hooks on the shared model, so predictions run one at a time
CMD gunicorn -b 0.0.0.0:${PORT} -w 1 --threads 2 --timeout 120 app:app
