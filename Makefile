devices ?= 0
force-build ?= false
use-gpu ?= true

ifeq ($(force-build),true)
	method = run
else
	method = no-build-run
endif

ifeq ($(use-gpu),true)
	gpu-options = --gpus '"device=$(devices)"'
else
	gpu-options =
endif

build:
	DOCKER_BUILDKIT=1 docker build -t tidl-toy-docker-image:1.0 .

no-build-run:
	docker run -it -d --shm-size=4096m \
	--network host $(gpu-options) \
	--mount type=bind,source="$(shell pwd)"/assets,target=/home/workdir/assets \
	--name tidl-toy-docker-container tidl-toy-docker-image:1.0

run: build no-build-run

start:
	make $(method)

exec:
	docker exec -it tidl-toy-docker-container /bin/bash

stop:
	docker stop tidl-toy-docker-container
	docker rm tidl-toy-docker-container
