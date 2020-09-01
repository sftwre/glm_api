FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# only copy necessary files
COPY customer_segmentation_model.pkl .

COPY features.pkl .

COPY api.py .

CMD [ "python", "./api.py" ]