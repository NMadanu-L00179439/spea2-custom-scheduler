FROM python:3.9-slim

WORKDIR /app

COPY spea2_scheduler.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "spea2_scheduler.py"]
