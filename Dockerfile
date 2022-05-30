FROM ubuntu

RUN apt update
RUN apt install python3-pip -y
RUN pip3 install Flask
RUN pip3 install flask-sqlalchemy
RUN pip3 install flask-login
RUN pip3 install flask-bcrypt
RUN pip3 install flask-wtf
RUN pip3 install transformers
RUN pip3 install nltk
RUN pip3 install torch
RUN pip3 install sklearn
RUN pip3 install sentence_transformers
RUN pip3 install -U flask-paginate

WORKDIR /app

COPY . .

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]