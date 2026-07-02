import os
from functools import lru_cache

class Settings:
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "1000"))

@lru_cache
def get_settings() -> Settings:
    return Settings()