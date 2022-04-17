FROM python:3.8-alpine

EXPOSE 5000/tcp

WORKDIR /

COPY requirements.txt
COPY xgboost wheel/xgboost-1.5.1-cp38-cp38-win32.whl

RUN pip install xgboost-1.5.1-cp38-cp38-win32.whl
RUN pip install -r requirements.txt

COPY flask_app.py
COPY rf.model
COPY svm.model
COPY xgb.model