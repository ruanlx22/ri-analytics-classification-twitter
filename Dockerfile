FROM sellpy/python3-jupyter-sklearn-java

RUN pip3 install --upgrade pip
RUN pip3 install flask==1.0.2
RUN pip3 install numpy==1.15.4
RUN pip3 install scipy==1.1.0
RUN pip3 install spacy==2.0.18
RUN pip3 install scikit-learn==0.20.1

RUN python3 -m spacy download it
RUN python3 -m spacy download en

RUN pip3 install gdown
RUN apt-get install unzip

ARG GDRIVE_DL_LINK

RUN gdown https://drive.google.com/uc?id=${GDRIVE_DL_LINK}
# Add local files and folders
ADD / /app/amazon-kinesis-client-python/

RUN unzip -o models.zip

EXPOSE 9655

CMD [ "python3", "./starter.py" ]