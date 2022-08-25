FROM jupyter/scipy-notebook

RUN mkdir models

ENV MODEL_PATH=/home/jovyan/models

RUN pip install joblib

COPY data.csv ./data.csv
COPY runner.py ./runner.py

RUN python runner.py


