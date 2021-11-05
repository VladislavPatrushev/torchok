from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins import TrainingTypePlugin
from typing_extensions import Literal


class StructureParams(BaseModel):
    name: str
    params: dict = {}
    aux_params: Optional[dict] = {}


class DataLoaderParams(BaseModel):
    batch_size: int
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = False
    use_custom_collate_fn: bool = False
    use_custom_batch_sampler: bool = False


class DatasetParams(BaseModel):
    name: str
    params: dict = {}
    transform: List[StructureParams]
    augment: List[StructureParams] = None
    dataloader_params: DataLoaderParams


class DataParams(BaseModel):
    common_params: Optional[dict] = {}
    train_params: DatasetParams
    valid_params: DatasetParams
    test_params: Optional[DatasetParams]


class TrainerParams(BaseModel):
    enable_checkpointing: bool = True
    default_root_dir: Optional[str] = None
    gradient_clip_val: Optional[Union[int, float]] = None
    gradient_clip_algorithm: Optional[str] = None
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    devices: Optional[Union[List[int], str, int]] = None
    gpus: Optional[Union[List[int], str, int]] = None
    auto_select_gpus: bool = False
    tpu_cores: Optional[Union[List[int], str, int]] = None
    ipus: Optional[int] = None
    enable_progress_bar: bool = True
    use_rich_bar: bool = False
    refresh_rate_per_second: Optional[int] = 10
    overfit_batches: Union[int, float] = 0.0
    track_grad_norm: Union[int, float, str] = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: Union[int, bool] = False
    accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0
    limit_predict_batches: Union[int, float] = 1.0
    val_check_interval: Union[int, float] = 1.0
    flush_logs_every_n_steps: Optional[int] = None
    log_every_n_steps: int = 50
    accelerator: Optional[Union[str, Accelerator]] = None
    strategy: Optional[Union[str, TrainingTypePlugin]] = None
    sync_batchnorm: bool = False
    precision: Union[int, str] = 32
    enable_model_summary: bool = True
    weights_summary: Optional[str] = "top"
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Optional[Union[Path, str]] = None
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Union[bool, str] = False
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: Union[str, bool] = False
    prepare_data_per_node: Optional[bool] = None
    amp_backend: str = "native"
    amp_level: Optional[str] = None
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = "max_size_cycle"
    stochastic_weight_avg: bool = False
    terminate_on_nan: Optional[bool] = None

    class Config:
        arbitrary_types_allowed = True


class TensorboardLoggerParams(BaseModel):
    logger: Literal['tensorboard']
    log_graph: bool = False


class AWSSecretsManagerParams(BaseModel):
    region: str
    mlflow_secret: str


class MLFlowLoggerParams(BaseModel):
    logger: Literal['mlflow']
    tracking_uri: Optional[str] = None
    tags: Optional[Dict[str, Any]] = {}
    save_dir: Optional[str] = './mlruns'
    secrets_manager: Optional[AWSSecretsManagerParams] = None


class CheckpointParams(BaseModel):
    filename: Optional[str] = None
    monitor: str = 'valid/loss'
    verbose: bool = False
    save_last: bool = False
    save_top_k: Optional[int] = 1
    save_weights_only: bool = False
    mode: str = "min"
    auto_insert_metric_name: bool = False
    every_n_train_steps: Optional[int] = None
    every_n_val_epochs: Optional[int] = None


class ProfilerParams(BaseModel):
    name: str = 'simple'
    save_profile: bool = False


class LossParams(BaseModel):
    loss_list: List[StructureParams]
    weights: List[float] = None
    log_separate_losses: bool = False


class MetricParams(BaseModel):
    name: str
    params: dict = {}
    phases: List[Union[Literal['train'], Literal['valid'], Literal['test']]] = ['train', 'valid', 'test']


class TrainConfigParams(BaseModel, Namespace):
    task: StructureParams
    loss: LossParams
    optimizers: Union[StructureParams, Dict[str, StructureParams]]
    schedulers: Union[StructureParams, Dict[str, StructureParams]] = None
    data: DataParams
    metrics: List[MetricParams] = []
    trainer: TrainerParams
    logger: Union[TensorboardLoggerParams, MLFlowLoggerParams] = \
        Field(..., descriminator='logger')
    experiment_name: str
    log_dir: str = './logs'
    checkpoint: CheckpointParams
    callbacks: Optional[List[StructureParams]] = []
    profiler: Optional[ProfilerParams]
