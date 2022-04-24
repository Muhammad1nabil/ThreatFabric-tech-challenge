FROM python:3.8-alpine

EXPOSE 5000/tcp

WORKDIR /

COPY requirements.txt .
RUN pip install --upgrade pip==21.0.1 certifi==2020.12.5 idna==2.10 python_dateutil==2.8.1 pyrsistent==0.17.3 chardet==3.0.4 requests==2.24.0 pynacl==1.4.0 cryptography==3.3.1
RUN pip install -r requirements.txt

COPY flask_app.py .
COPY wsgi.py .
COPY rf.model .
COPY svm.model .
COPY xgb.model .