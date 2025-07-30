FROM python:3.14.0rc1

RUN pip install mlflow azure-storage-blob numpy scipy pandas scikit-learn cloudpickle

COPY train.py .
COPY wine-quality.csv .
