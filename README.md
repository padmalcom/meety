# meety

## Run with Poetry:
Install [poetry](https://python-poetry.org/) (virtual environment, dependency and package manager). 

First time set up of virtual environment:

`poetry install (only for setting it up first time)`

Run script:

`poetry run sh run.sh`

## Required models:

From [Vosk](https://alphacephei.com/vosk/models):

- vosk-model-small-de-0.15

- vosk-model-de-0.21

- vosk-model-spk-0.4

- vosk-recasepunc-de-0.21

From [Huggingface](https://huggingface.co):

 - [T-Systems-onsite/mt5-small-sum-de-en-v2](https://huggingface.co/T-Systems-onsite/mt5-small-sum-de-en-v2)
 - [dbmdz/bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased/tree/main)

Download all models as described and put them into folder `model`. Drop prefix folders if necessary (e.g. 
`dbmdz/bert-base-german-uncased` --> `model/bert-base-german-uncased`).