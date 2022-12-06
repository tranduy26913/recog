FROM python:3.7.10-slim

WORKDIR /app
COPY . .

RUN apt-get update
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["streamlit", "run", "app.py"]