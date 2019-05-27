FROM petronetto/docker-python-deep-learning:v-26 as builder

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install setuptools==37.0.0
RUN pip wheel -r requirements.txt -w /wheels

FROM petronetto/docker-python-deep-learning:v-26

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt --find-links /wheels

COPY . /app
ENV PYTHONPATH /app
CMD [ "python", "-u", "./server.py" ]