# Use the official Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy only the files needed for installing dependencies
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install --no-cache-dir poetry

# Configure Poetry: Do not create a virtual environment inside the container
RUN poetry config virtualenvs.create false

# Install dependencies using Poetry, skipping development dependencies
RUN poetry install

COPY . /app

# Run the application
CMD ["python", "run_flow.py"]