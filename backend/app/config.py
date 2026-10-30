from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Plant Disease Detection"
    environment: str = "development"
    cpu_threads: int = 2
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'plant_disease.db'}"
    model_dir: Path = PROJECT_ROOT / "saved_models"
    report_dir: Path = PROJECT_ROOT / "reports" / "generated"
    max_upload_size_bytes: int = 10 * 1024 * 1024

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PLANT_", extra="ignore")


settings = Settings()
