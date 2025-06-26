# --- Build Stage ---
# Use a slim Python image for the build environment
FROM python:3.9-slim as builder

# Set the working directory
WORKDIR /app

# Install system dependencies required for opencv-python and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# First, copy only the requirements file to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
# Use a fresh, slim Python image for the final image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy necessary application files
COPY app.py .
COPY backend/ ./backend/
COPY templates/ ./templates/
COPY logo.png .
COPY MLI.png .

# Expose the port the app runs on
EXPOSE 5001

# Set the command to run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"] 