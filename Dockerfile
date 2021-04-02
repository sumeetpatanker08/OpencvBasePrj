FROM python:3.9

ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt

RUN apt-get update ##[edited]

RUN apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]

CMD ["app.py"]