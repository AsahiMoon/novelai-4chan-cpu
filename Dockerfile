FROM python:3.10

WORKDIR /novelai
ADD program /novelai

RUN pip install virtualenv 
RUN ./setup.sh