FROM python:3.10-slim

# Install system dependencies (including libc6-dev for additional headers)
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

# Set working directory
WORKDIR /ChatAnalysis

# Copy all project files into the container
COPY . .

# Upgrade pip to get the latest binary wheels
RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit’s port
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501

# Adjust permissions if needed
RUN chmod -R 777 /ChatAnalysis