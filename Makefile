build:
	docker build -t orf .

train:
	docker run \
		-d \
		--env-file ${ENV_FILE} \
		-v $(pwd)/data:/app/data \
		-v $(pwd)/model:/app/model \
		orf python train.py

server:
	docker run \
		-d \
		--env-file ${ENV_FILE} \
		-v $(pwd)/data:/app/data \
		-v $(pwd)/model:/app/model \
		orf python server.py