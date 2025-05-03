# Use official Python image
FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y gcc

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

EXPOSE 5000

CMD ["python", "m.py"]
