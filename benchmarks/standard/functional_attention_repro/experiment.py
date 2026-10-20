import random
import numpy as np
import torch
import wandb
import time
from typing import Optional, Dict, Any

def generate_seed() -> int:
    return int(time.time() * 1000) % (2**31) + random.randint(0, 1000)

def seed_everything(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class ExperimentLogger:
    def __init__(
        self, 
        project_name: str,
        run_name: str,
        config: Dict[str, Any],
        tags: Optional[list] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.run_name = run_name
        
        if self.enabled:
            wandb.init(
                entity="FuncAttn",
                project=project_name,
                name=run_name,
                config=config,
                tags=tags or []
            )
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def log_summary(self, summary: Dict[str, Any]):
        if self.enabled:
            for key, value in summary.items():
                wandb.run.summary[key] = value
    
    def finish(self):
        if self.enabled:
            wandb.finish()


class TrainingTimer:
    def __init__(self):
        self.total_time = 0.0
        self.epoch_start = None
    
    def start_epoch(self):
        torch.cuda.synchronize()
        self.epoch_start = time.time()
    
    def end_epoch(self):
        torch.cuda.synchronize()
        epoch_time = time.time() - self.epoch_start
        self.total_time += epoch_time
        return epoch_time
    
    def get_avg_epoch_time(self, num_epochs):
        return self.total_time / num_epochs if num_epochs > 0 else 0


def get_memory_usage():
    return torch.cuda.max_memory_allocated() / (1024**2)