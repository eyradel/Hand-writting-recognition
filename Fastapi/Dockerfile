
FROM python:3.9-slim


WORKDIR /app


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


RUN pip list
RUN ls /usr/local/bin


COPY . .


EXPOSE 8006


ENV PYTHONUNBUFFERED True


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]
