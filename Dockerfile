FROM python:3.10-slim

WORKDIR /app

# Install system utilities & AWS CLI
RUN apt-get update && apt-get install -y --no-install-recommends     curl     unzip     build-essential     && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"     && unzip awscliv2.zip     && ./aws/install     && rm -rf awscliv2.zip ./aws /var/lib/apt/lists/*

COPY requirements.txt requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY . /app

EXPOSE 8000

CMD ["python3", "app.py"]
