FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

COPY *.pkl .

COPY exercise_26_test.csv .

COPY api.py .

COPY tests .

CMD [ "python", "./api.py", '-ip localhost', '-port 8080']