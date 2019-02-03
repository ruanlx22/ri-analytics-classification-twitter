FROM sellpy/python3-jupyter-sklearn-java

RUN pip install --upgrade pip
RUN pip install flask
RUN pip install numpy
RUN pip install scipy
RUN pip install spacy
RUN pip install scikit-learn==0.20.1

RUN python3 -m spacy download it
RUN python3 -m spacy download en

RUN pip install gdown
RUN apt-get install unzip

ARG GDRIVE_DL_LINK

RUN gdown https://drive.google.com/uc?id=${GDRIVE_DL_LINK}

# Add local files and folders
ADD / /app/amazon-kinesis-client-python/
RUN unzip -o models.zip

EXPOSE 9655

CMD [ "python3", "./starter.py" ]