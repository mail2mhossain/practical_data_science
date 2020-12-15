# Defining base image
FROM python:3.8.2-slim

# Installing packages from PyPi
RUN pip install --upgrade pip && \
    pip install scipy==1.5.2 && \
    pip install numpy==1.18.5 && \
    pip install matplotlib==3.3.2 && \
    pip install pandas==1.1.3 && \
    pip install scikit-learn==0.23.2 && \
    pip install xgboost==1.2.1 && \
    pip install mlflow==1.12.1 && \
    pip install psycopg2-binary==2.8.6 && \
    pip install boto3==1.16.28

# Defining start up command
EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]
