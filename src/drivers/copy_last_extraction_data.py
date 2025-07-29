import shutil
import subprocess
import argparse

from config import LAST_RUN_PATH

CONTAINER_NAME = "pdf_metadata_extraction_worker"
CONTAINER_DATA_PATH = f"/app/{LAST_RUN_PATH.parent.name}/{LAST_RUN_PATH.name}"


def copy_last_extraction_data(remote=None):
    if remote:
        # Remote copy logic
        remote_tmp_dir = f"/tmp/{LAST_RUN_PATH.name}_copy"
        # Remove remote temp dir if exists
        subprocess.run(["ssh", remote, f"rm -rf {remote_tmp_dir}"], check=True)
        # Docker cp on remote
        subprocess.run(
            ["ssh", remote, f"docker cp {CONTAINER_NAME}:{CONTAINER_DATA_PATH} {remote_tmp_dir}"],
            check=True,
        )
        # Remove local LAST_RUN_PATH if exists
        shutil.rmtree(LAST_RUN_PATH, ignore_errors=True)
        # Copy from remote to local
        subprocess.run(["scp", "-r", f"{remote}:{remote_tmp_dir}", str(LAST_RUN_PATH)], check=True)
        # Clean up remote temp dir
        subprocess.run(["ssh", remote, f"rm -rf {remote_tmp_dir}"], check=True)
        print(f"Copied last extraction data from {remote}:{remote_tmp_dir} to {LAST_RUN_PATH}")
    else:
        if LAST_RUN_PATH.exists() and LAST_RUN_PATH.is_dir() and any(LAST_RUN_PATH.iterdir()):
            print(f"Last run data already exists at {LAST_RUN_PATH}")
            return
        else:
            shutil.rmtree(LAST_RUN_PATH, ignore_errors=True)
        subprocess.run(["docker", "cp", f"{CONTAINER_NAME}:{CONTAINER_DATA_PATH}", LAST_RUN_PATH], check=True)
        print(f"Copied last extraction data locally to {LAST_RUN_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy last extraction data from local or remote Docker container.")
    parser.add_argument("--remote", type=str, default=None, help="Remote server in user@host format")
    args = parser.parse_args()
    copy_last_extraction_data(remote=args.remote)
