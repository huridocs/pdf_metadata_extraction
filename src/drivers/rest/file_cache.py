import json
import gzip
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
from config import MODELS_DATA_PATH


class FileCacheManager:
    """File-based cache manager for large samples data with compression"""

    def __init__(self):
        self.cache_dir = Path(MODELS_DATA_PATH, "cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_ttl = 86400  # 1 day in seconds

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Generate cache file path with hashed key to avoid filesystem issues"""
        # Hash the key to avoid filesystem path length/character issues
        key_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key_hash}.cache.gz"

    def _is_cache_valid(self, file_path: Path) -> bool:
        """Check if cache file exists and is not expired"""
        if not file_path.exists():
            return False

        # Check if file is older than TTL
        file_age = time.time() - file_path.stat().st_mtime
        return file_age < self.cache_ttl

    def _compress_and_save(self, data: Any, file_path: Path) -> None:
        """Compress and save data to cache file"""
        json_str = json.dumps(data)
        json_bytes = json_str.encode("utf-8")

        with gzip.open(file_path, "wb") as f:
            f.write(json_bytes)

    def _load_and_decompress(self, file_path: Path) -> Any:
        """Load and decompress data from cache file"""
        with gzip.open(file_path, "rb") as f:
            json_bytes = f.read()

        json_str = json_bytes.decode("utf-8")
        return json.loads(json_str)

    def get_cached_samples(self, cache_key: str) -> Optional[Any]:
        """Get cached samples data if valid"""
        try:
            file_path = self._get_cache_file_path(cache_key)

            if self._is_cache_valid(file_path):
                return self._load_and_decompress(file_path)

            return None
        except Exception:
            # If cache reading fails, return None to fallback to database
            return None

    def cache_samples(self, cache_key: str, data: Any) -> None:
        """Cache samples data with compression"""
        try:
            file_path = self._get_cache_file_path(cache_key)
            self._compress_and_save(data, file_path)
        except Exception:
            # If caching fails, just continue without caching
            pass

    def cleanup_expired_cache(self) -> None:
        """Remove expired cache files"""
        try:
            current_time = time.time()
            for cache_file in self.cache_dir.glob("*.cache.gz"):
                if current_time - cache_file.stat().st_mtime > self.cache_ttl:
                    cache_file.unlink(missing_ok=True)
        except Exception:
            # If cleanup fails, just continue
            pass

    def delete_cache(self, cache_key: str) -> bool:
        """Delete cache file for a specific cache key"""
        try:
            file_path = self._get_cache_file_path(cache_key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            # If deletion fails, return False
            return False

    @staticmethod
    def get_training_cache_key(run_name: str, extraction_name: str) -> str:
        """Generate cache key for training samples"""
        return f"samples_training:{run_name}:{extraction_name}"

    @staticmethod
    def get_prediction_cache_key(run_name: str, extraction_name: str) -> str:
        """Generate cache key for prediction samples"""
        return f"samples_prediction:{run_name}:{extraction_name}"


file_cache_manager = FileCacheManager()
