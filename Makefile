install:
	. venv/bin/activate; pip install -Ur requirements.txt

activate:
	. venv/bin/activate

install_venv:
	python3 -m venv venv
	. venv/bin/activate; python -m pip install --upgrade pip
	. venv/bin/activate; python -m pip install -r dev-requirements.txt

formatter:
	. venv/bin/activate; command black --line-length 125 .

check_format:
	. venv/bin/activate; command black --line-length 125 . --check

test:
	. venv/bin/activate; command cd src; command pytest

remove_docker_containers:
	docker compose ps -q | xargs docker rm

remove_docker_images:
	docker compose config --images | xargs docker rmi

start:
	docker compose -f local-docker-compose.yml up --attach pdf_metadata_extraction_worker --attach pdf_metadata_extraction_api --build

start_gpu:
	docker compose -f gpu-docker-compose.yml up --attach pdf_metadata_extraction_worker --attach pdf_metadata_extraction_api --build


start_local_gpu:
	docker compose -f local-gpu-docker-compose.yml up --attach pdf_metadata_extraction_worker --attach pdf_metadata_extraction_api --build

start_detached:
	docker compose up --build -d

start_for_testing:
	docker compose up --attach pdf_metadata_extraction_worker --attach pdf_metadata_extraction_api --build

stop:
	docker compose stop

delete_queues:
	. venv/bin/activate; python scripts/delete_queues.py

download_models:
	. venv/bin/activate; command cd src; python download_models.py

free_up_space:
	df -h
	sudo rm -rf /usr/share/dotnet
	sudo rm -rf /opt/ghc
	sudo rm -rf "/usr/local/share/boost"
	sudo rm -rf "$AGENT_TOOLSDIRECTORY"
	sudo apt-get remove -y '^llvm-.*' || true
	sudo apt-get remove -y 'php.*' || true
	sudo apt-get remove -y google-cloud-sdk hhvm google-chrome-stable firefox mono-devel || true
	sudo apt-get autoremove -y
	sudo apt-get clean
	sudo rm -rf /usr/share/dotnet
	sudo rm -rf /usr/local/lib/android
	sudo rm -rf /opt/hostedtoolcache/CodeQL
	sudo docker image prune --all --force
	df -h

