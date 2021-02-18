FROM python:3.7
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY hw1.pkl ./
CMD ["python", "app.py"]
EXPOSE 5000
