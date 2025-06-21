FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies required for popular Python data science and plotting libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libgdk-pixbuf2.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libpango-1.0-0 \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    python3-tk \
    libc6-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /ChatAnalysis


COPY dspy_agent/requirements_nicegui_dspy.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_nicegui_dspy.txt

COPY . .

EXPOSE 9090
