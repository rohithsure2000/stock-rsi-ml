FROM python:3.12-slim

LABEL maintainer="Rohith Sure <surerh2000@gmail.com>"
LABEL description="RSI + Machine Learning stock movement prediction pipeline"

WORKDIR /app

# System deps for matplotlib (fonts) kept minimal on purpose
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed outputs/figures

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--ticker", "AAPL", "--verbose"]
