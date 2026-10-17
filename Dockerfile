FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MMA_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app
COPY requirements.lock pyproject.toml ./
COPY src ./src
COPY apps ./apps
COPY configs ./configs
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps .

EXPOSE 8501
CMD ["streamlit", "run", "apps/streamlit_app.py", "--server.address=0.0.0.0"]
