FROM python:3.13-slim

# System dependencies for scientific Python stack
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libstdc++6 \
    libgl1 \
    libglib2.0-0 \
    liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    libhdf5-dev \
    libfftw3-dev \
    libcfitsio-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements
WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

CMD ["python", "main.py"]

