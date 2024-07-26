parse_pdf:
	python utils/text_retrieval.py

build_docker:
	docker build -t chatbot .

run_docker:
	docker run -ti --env-file .env chatbot
