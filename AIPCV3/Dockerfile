# we start from a lightweight python 3.9 image
FROM python:3.9-slim

# we update the package list and install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# we set the working directory
WORKDIR /app

# we copy requirements and install them
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# we copy all project files into the container
COPY . .

# we expose port 4000 for our fastapi app
EXPOSE 4000

# we set the container’s default startup command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4000"]
