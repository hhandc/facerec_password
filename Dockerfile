FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y cmake gcc g++ make

# Create a Python environment
RUN conda create -n face_env python=3.11 -y

# Activate the environment and install dependencies
RUN echo "conda activate face_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/face_env/bin:$PATH

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your app code
COPY . /app
WORKDIR /app

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
