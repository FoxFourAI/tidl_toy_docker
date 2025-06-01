devices ?= 0
force-build ?= false
use-gpu ?= true
image-name ?= tidl-toy-docker-image:1.0
container-name ?= tidl-toy-docker-container

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
	DOCKER_BUILDKIT=1 docker build -t $(image-name) .

no-build-run:
	docker run -it -d --shm-size=4096m \
	--network host $(gpu-options) \
	--mount type=bind,source="$(shell pwd)"/assets,target=/home/workdir/assets \
	--name $(container-name) $(image-name)

run: build no-build-run

start:
	make $(method)

exec:
	docker exec -it $(container-name) /bin/bash

stop:
	docker stop $(container-name)
	docker rm $(container-name)
