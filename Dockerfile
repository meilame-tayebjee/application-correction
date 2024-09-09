FROM ubuntu:22.04
WORKDIR ${HOME}/titanic
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
RUN apt-get install -y git
# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY api ./api
COPY docs ./docs

RUN ["chmod", "+x", "./api/run.sh"]
CMD ["bash", "-c", "./api/run.sh"]