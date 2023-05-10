import os
import torch
import random
import numpy as np
import wandb
import hydra
from hydra.utils import instantiate
import omegaconf
from omegaconf import open_dict
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from trainers.hipt_trainer import HiPTTrainer
from trainers.fcp_trainer import FCPTrainer
from trainers.population_trainer import SPPopulationTrainer
from evaluation.evals import eval_rollout, plot_sp_heatmap


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    run_name = f"{config.gym_id}__{config.layout.name}__{config.model.name}__{config.seed}__{os.getcwd().split('/')[-2]}__{os.getcwd().split('/')[-1]}"
    
    if config.logging.track:        
        config_dict = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project=config.logging.wandb_project_name,
            entity=config.logging.wandb_entity,
            sync_tensorboard=True,
            config = config_dict,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        wandb.run.log_code(".")
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    torch.backends.cudnn.deterministic = config.torch_deterministic
    trainer = instantiate(config.model.trainer, config)
    
    if config.train:
        trainer.train()

    if config.eval:
        if isinstance(trainer, HiPTTrainer) or isinstance(trainer, FCPTrainer):
            eval_rollout(trainer, config)
        elif isinstance(trainer, SPPopulationTrainer):
            plot_sp_heatmap(trainer, config)
            
if __name__ == "__main__":
    main()
