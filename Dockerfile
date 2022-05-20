FROM ubuntu

RUN apt update
RUN apt install python-pip -y
RUN pip install Flask

WORKDIR /app

COPY . .

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]