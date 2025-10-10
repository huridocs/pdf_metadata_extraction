import json
import gzip
import shutil
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from config import MODELS_DATA_PATH, LAST_RUN_PATH


class SamplesCacheUseCase:
    def __init__(self):
        self.cache_dir = Path(MODELS_DATA_PATH, "cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_ttl = 86400

    def _get_cache_file_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.cache.gz"

    def _is_cache_valid(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False
        file_age = time.time() - file_path.stat().st_mtime
        return file_age < self.cache_ttl

    def _compress_and_save(self, data: list[BaseModel], file_path: Path) -> None:
        serialized_data = [item.model_dump() for item in data]
        json_str = json.dumps(serialized_data)
        json_bytes = json_str.encode("utf-8")
        with gzip.open(file_path, "wb") as f:
            f.write(json_bytes)

    def _load_and_decompress(self, file_path: Path) -> list[dict]:
        with gzip.open(file_path, "rb") as f:
            json_bytes = f.read()
        json_str = json_bytes.decode("utf-8")
        return json.loads(json_str)

    def get_cached_samples(self, cache_key: str) -> Optional[list[dict]]:
        try:
            file_path = self._get_cache_file_path(cache_key)
            if self._is_cache_valid(file_path):
                return self._load_and_decompress(file_path)
            return None
        except Exception:
            return None

    def cache_samples(self, cache_key: str, data: list[BaseModel]) -> None:
        try:
            file_path = self._get_cache_file_path(cache_key)
            self._compress_and_save(data, file_path)

            if "train" in cache_key:
                self._save_last_run_training_data(data)
        except Exception:
            pass

    def _save_last_run_training_data(self, data: list[BaseModel]) -> None:
        try:
            last_run_dir = Path(LAST_RUN_PATH)

            if last_run_dir.exists():
                shutil.rmtree(last_run_dir)

            last_run_dir.mkdir(exist_ok=True, parents=True)

            metadata_file = last_run_dir / "metadata.json"
            metadata = {"timestamp": time.time(), "sample_count": len(data)}
            metadata_file.write_text(json.dumps(metadata, indent=2))

            data_file = last_run_dir / "training_data.json.gz"
            self._compress_and_save(data, data_file)
        except Exception:
            pass

    def cleanup_expired_cache(self) -> None:
        try:
            current_time = time.time()
            for cache_file in self.cache_dir.glob("*.cache.gz"):
                if current_time - cache_file.stat().st_mtime > self.cache_ttl:
                    cache_file.unlink(missing_ok=True)
        except Exception:
            pass

    def delete_cache(self, cache_key: str) -> bool:
        try:
            file_path = self._get_cache_file_path(cache_key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def get_training_cache_key(run_name: str, extraction_name: str) -> str:
        return f"samples_training_{run_name}_{extraction_name}"

    @staticmethod
    def get_prediction_cache_key(run_name: str, extraction_name: str) -> str:
        return f"samples_prediction_{run_name}_{extraction_name}"
