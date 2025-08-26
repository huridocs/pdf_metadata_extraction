install:
	. .venv/bin/activate; pip install -Ur requirements.txt

activate:
	. .venv/bin/activate

install-venv:
	python3 -m venv .venv
	. .venv/bin/activate; uv pip install --upgrade pip
	. .venv/bin/activate; uv pip install -r dev-requirements.txt

formatter:
	. .venv/bin/activate; command black --line-length 125 .

check-format:
	. .venv/bin/activate; command black --line-length 125 . --check

test:
	. .venv/bin/activate; command cd src; command python -m pytest -n 0 tests/test_app.py tests/test_end_to_end.py tests/test_end_to_end_paragraph_extractor.py

test-cloud:
	. .venv/bin/activate; command cd src; command python -m pytest -n 0 tests/test_end_to_end.py

wait-for-queues:
	. .venv/bin/activate; command cd scripts; command python wait_for_queues.py

remove-docker-containers:
	docker compose ps -q | xargs docker rm

remove-docker-images:
	docker compose config --images | xargs docker rmi

start-test:
	docker compose up --attach pdf_metadata_extraction_worker --attach pdf_metadata_extraction_api --build

start-windows:
	docker compose -f windows-gpu-docker-compose.yml up --attach pdf_metadata_extraction_worker --attach pdf_metadata_extraction_api --build

start-detached:
	docker compose up --build -d

start:
	docker compose -f gpu-docker-compose.yml up --attach pdf_metadata_extraction_worker --attach pdf_metadata_extraction_api --build


start-no-gpu:
	docker compose up --attach pdf_metadata_extraction_worker --attach pdf_metadata_extraction_api --build

stop:
	docker compose stop

delete-queues:
	. .venv/bin/activate; python scripts/delete_queues.py

gpu:
	. .venv/bin/activate; command cd src; python is_gpu_available.py

free-up-space:
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

