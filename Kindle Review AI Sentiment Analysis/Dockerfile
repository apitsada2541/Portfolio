# Use the official Python image from Docker Hub (matching your Python version)
FROM python:3.12-slim

# Set bash as the default shell
SHELL ["/bin/bash", "-c"]

# Install system dependencies and utilities (for better setup of the environment)
RUN apt-get update -qq && apt-get upgrade -qq && \
    apt-get install -qq man wget sudo vim tmux

# Upgrade pip to the latest version
RUN yes | pip install --upgrade pip

# Copy requirements.txt and install dependencies from it
COPY requirements.txt /home/
RUN pip install --no-cache-dir -r /home/requirements.txt

# Clean up pip cache to reduce the image size
RUN pip cache purge

# Copy project files (your notebooks, pickle file, .env file, etc.)
COPY agents.py /home/
COPY Stitching_GenAI.ipynb /home/
COPY WebApp.py /home/

# Set the working directory to /home (where your project files are located)
WORKDIR /home
