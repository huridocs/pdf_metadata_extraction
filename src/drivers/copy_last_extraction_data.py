import shutil
import subprocess

from config import LAST_RUN_PATH

CONTAINER_NAME = "pdf_metadata_extraction_worker"
CONTAINER_DATA_PATH = f"/app/{LAST_RUN_PATH.parent.name}/{LAST_RUN_PATH.name}"


def copy_last_extraction_data():
    if LAST_RUN_PATH.exists() and LAST_RUN_PATH.is_dir() and any(LAST_RUN_PATH.iterdir()):
        print(f"Last run data already exists at {LAST_RUN_PATH}")
        return
    else:
        shutil.rmtree(LAST_RUN_PATH, ignore_errors=True)

    subprocess.run(["docker", "cp", f"{CONTAINER_NAME}:{CONTAINER_DATA_PATH}", LAST_RUN_PATH], check=True)


if __name__ == "__main__":
    copy_last_extraction_data()
