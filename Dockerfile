FROM ubuntu:16.04

MAINTAINER HoangPH <hoangphan0710@gmail.com>

RUN apt-get update
RUN apt-get -y install cron
RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
RUN set -xe \
    && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt

COPY . /code

RUN chmod a+x /code/retrain.sh

EXPOSE 9000

# Run the command on container startup
CMD make guni
