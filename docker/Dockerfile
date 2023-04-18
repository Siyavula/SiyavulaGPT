FROM python:3.10.11-bullseye as localdev

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

COPY src src

WORKDIR src

RUN cp examples/authentication-methods/no-auth/main.py server/main.py

RUN pip install --upgrade pip \
    poetry

RUN poetry config virtualenvs.create false && \
    poetry install --no-dev

ENTRYPOINT ["poetry", "run", "start"]