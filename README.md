# rag_from_pdf
A simple chatbot capable of answering GDPR related questions based on `GDPR Art 1-21.pdf` file with CL interface. This version leverages OpenAI API for language model (`gpt-3.5-turbo-0125`) and FAISS for similarity search with OpenAI embeddings (`text-embedding-3-large`). You can replace these models easily with other OpenAI models by defining them as environment variables. You can check `Summary.ipynb` notebook for more detailed description of the system.

## Table of contents
- [Pre-requirements](#pre-requirements)
- [Installation](#installation)
- [Data extraction from PDF](#data-extraction-from-pdf)

## Pre-requirements
You need [Docker](https://www.docker.com/) to be able to run containers with the application. Additionally it's preferable to have [make](https://www.gnu.org/software/make/) for easy project installation.

## Installation
Create an `.env` file or manually input environment variables like below for correct project functioning. Don't forget to put your own OpenAI API key.
```
PATH_TO_PDF=files/GDPR Art 1-21.pdf
PATH_TO_SUMMARIES=files/summaries.txt
PATH_FOR_DOCS=files/documents.json
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_MODEL=gpt-3.5-turbo-0125
EMBEDDING_MODEL=text-embedding-3-large
```

There are two ways to run the project for usage.

1. The easiest way is to use `make` and simply run the commands:
```
make build_docker
make run_docker
```

2. If you don't have `make`, then you can run docker commands itself.
```
docker build -t chatbot .
docker run -ti --env-file .env chatbot
```

## Data extraction from PDF
All the necessary data is already processed and placed to `files` folder for simplicity. If you want to run data processing again and to recreate `files/documents.json` file on your own, run one of the following script depending on whether you have `make`:
```
make parse_pdf
```

or

```
python utils/text_retrieval.py
```
Don't forget to define environment variables, for data extraction you need to define at least the directory where PDF file, txt file with summaries are stored and where you want to save the final json with documents. See the example below:
```
PATH_TO_PDF=files/GDPR Art 1-21.pdf
PATH_TO_SUMMARIES=files/summaries.txt
PATH_FOR_DOCS=files/documents.json
```
