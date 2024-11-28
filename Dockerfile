FROM python:3.9-slim

# Set the working directory
WORKDIR /ChatAnalysis

# Copy all project files into the container
COPY . .

# Update package lists and install required system dependencies
RUN apt-get update && apt-get install -y \
    libgdk-pixbuf2.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libpango-1.0-0 \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Make the working directory writable
RUN chmod -R 777 /ChatAnalysis

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Streamlit
EXPOSE 8501

# Set environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501

