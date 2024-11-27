FROM python:3.10

WORKDIR /ML_FastAPI_Docker_Heroku

COPY ./requirements.txt /ML_FastAPI_Docker_Heroku/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /ML_FastAPI_Docker_Heroku/requirements.txt

COPY ../app /ML_FastAPI_Docker_Heroku/app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
