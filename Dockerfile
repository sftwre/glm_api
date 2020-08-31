FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# only copy necessary files
COPY customer_segmentation_model.pkl .

COPY features.pkl .

COPY exercise_26_test.csv .

COPY api.py .

COPY tests .

CMD [ "python", "./api.py" ]