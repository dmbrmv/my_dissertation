"""Configuration management for TFT predictions."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Data configuration settings."""
    
    meteo_input: List[str] = Field(
        default=["prcp_e5l", "t_max_e5l", "t_min_e5l"],
        description="Meteorological input variables"
    )
    hydro_target: str = Field(
        default="q_mm_day",
        description="Hydrological target variable (q_mm_day or lvl_sm)"
    )
    static_parameters: List[str] = Field(
        default_factory=list,
        description="Static catchment parameters"
    )
    nc_variable: str = Field(
        default="nc_all_q",
        description="NetCDF variable name pattern"
    )
    index_col: str = Field(
        default="gauge_id",
        description="Index column name"
    )
    
    @validator('hydro_target')
    def validate_hydro_target(cls, v: str) -> str:
        """Validate hydrological target variable."""
        allowed_targets = ["q_mm_day", "lvl_sm", "lvl_mbs"]
        if v not in allowed_targets:
            raise ValueError(f"hydro_target must be one of {allowed_targets}")
        return v


class ModelConfig(BaseModel):
    """TFT model configuration settings."""
    
    input_chunk_length: int = Field(default=90, ge=1)
    output_chunk_length: int = Field(default=30, ge=1)
    hidden_size: int = Field(default=128, ge=1)
    lstm_layers: int = Field(default=2, ge=1)
    num_attention_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=1)
    n_epochs: int = Field(default=100, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    random_state: int = Field(default=42)
    likelihood: str = Field(default="quantile", regex="^(quantile|gaussian)$")
    quantiles: List[float] = Field(
        default=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    
    @validator('quantiles')
    def validate_quantiles(cls, v: List[float]) -> List[float]:
        """Validate quantile values."""
        if not all(0 <= q <= 1 for q in v):
            raise ValueError("All quantiles must be between 0 and 1")
        if 0.5 not in v:
            raise ValueError("Quantiles must include 0.5 (median)")
        return sorted(v)


class TrainingConfig(BaseModel):
    """Training configuration settings."""
    
    train_split: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_split: float = Field(default=0.2, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    early_stopping_patience: int = Field(default=10, ge=1)
    save_checkpoints: bool = Field(default=True)
    
    @validator('test_split')
    def validate_splits(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate that train + val + test splits sum to 1."""
        train = values.get('train_split', 0.7)
        val = values.get('val_split', 0.2)
        if abs(train + val + v - 1.0) > 1e-6:
            raise ValueError("train_split + val_split + test_split must equal 1.0")
        return v


class PathConfig(BaseModel):
    """Path configuration settings."""
    
    data_root: Path = Field(default=Path("../geo_data"))
    ws_geometry_file: Path = Field(default=Path("../geo_data/geometry/russia_ws.gpkg"))
    static_attributes_file: Path = Field(default=Path("../geo_data/attributes/static_with_height.csv"))
    nc_data_pattern: str = Field(default="../geo_data/ws_related_meteo/{nc_variable}/*.nc")
    output_dir: Path = Field(default=Path("outputs"))
    model_dir: Path = Field(default=Path("models"))
    
    @validator('data_root', 'output_dir', 'model_dir', pre=True)
    def validate_paths(cls, v: Path) -> Path:
        """Ensure paths are Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class Settings(BaseModel):
    """Main settings class containing all configuration."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "Settings":
        """Load settings from a YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, output_path: Path) -> None:
        """Save settings to a YAML file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)
    
    def update_static_parameters(self) -> None:
        """Update static parameters based on hydro_target."""
        base_parameters = [
            "for_pc_sse", "crp_pc_sse", "inu_pc_ult", "ire_pc_sse", "lka_pc_use",
            "prm_pc_sse", "pst_pc_sse", "cly_pc_sav", "slt_pc_sav", "snd_pc_sav",
            "kar_pc_sse", "urb_pc_sse", "gwt_cm_sav", "lkv_mc_usu", "rev_mc_usu",
            "sgr_dk_sav", "slp_dg_sav", "ws_area", "ele_mt_sav"
        ]
        
        if self.data.hydro_target in ["lvl_mbs", "lvl_sm"]:
            self.data.static_parameters = base_parameters + ["height_bs"]
            self.data.nc_variable = "nc_all_h"
        else:
            self.data.static_parameters = base_parameters
            self.data.nc_variable = "nc_all_q"


# Create default settings instance
default_settings = Settings()
default_settings.update_static_parameters()
