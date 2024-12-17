FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install --upgrade pip
RUN pip install sentencepiece tqdm unidecode

WORKDIR /workspaces/score-based-diffusion-lm

COPY . /workspaces/score-based-diffusion-lm
RUN pip install -e /workspaces/score-based-diffusion-lm

RUN useradd -m sdlm-user
USER sdlm-user