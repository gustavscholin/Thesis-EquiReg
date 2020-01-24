export USER_ID := $(shell id -u)
#export COMPOSE_CMD := docker-compose -f docker/docker-compose.yaml -H ssh://gpugpu.bahnhof.plattan.fi
export COMPOSE_CMD := docker-compose -f docker/docker-compose.yaml

check-env:
ifndef NVIDIA_VISIBLE_DEVICES
	$(error NVIDIA_VISIBLE_DEVICES environment variable is undefined)
endif

build:
	 $(COMPOSE_CMD) build
up: check-env
	 $(COMPOSE_CMD) up --detach
down:
	 $(COMPOSE_CMD) down
connect: up
	 ${COMPOSE_CMD} exec uda-thesis tmux new-session -As main
