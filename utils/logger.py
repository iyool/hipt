import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(os.getcwd() + "/summary")

    def log_train_info(self, log_dict):
        average_reward = np.mean([info["episode"]["r"] for info in log_dict["infos"]])

        self.writer.add_scalar("charts/avg_episodic_return", average_reward, log_dict["global_step"])
        self.writer.add_scalar("charts/learning_rate", log_dict["lr"], log_dict["global_step"])

        self.writer.add_scalar("losses/approx_kl", log_dict["approx_kl"], log_dict["global_step"])
        self.writer.add_scalar("losses/clipfrac", np.mean(log_dict["clipfracs"]), log_dict["global_step"])

        self.writer.add_scalar("charts/SPS", int(log_dict["global_step"] / (time.time() - log_dict["start_time"])), log_dict["global_step"])

        if self.config.model.name == "hipt":
            hi_rew = []
            for ind in range(self.config.training.num_envs):
                hi_rew.append(torch.sum(log_dict["hi_traj"][ind]["hi_rewards"]).item())
            avg_hi_rw = np.mean(hi_rew)

            self.writer.add_scalar("charts/avg_episodic_hi_rew", avg_hi_rw, log_dict["global_step"])

            if self.config.logging.track_kl_rew:
                self.writer.add_scalar("charts/avg_episodic_kl_div_rew", np.mean(log_dict["cum_kl_rew"]), log_dict["global_step"])
            
            self.writer.add_scalar("losses/value_loss_lo", log_dict["lo_v_loss"], log_dict["global_step"])
            self.writer.add_scalar("losses/value_loss_hi", log_dict["hi_v_loss"], log_dict["global_step"])
            self.writer.add_scalar("losses/policy_loss_lo", log_dict["lo_pg_loss"], log_dict["global_step"])
            self.writer.add_scalar("losses/policy_loss_hi", log_dict["hi_pg_loss"], log_dict["global_step"])
            self.writer.add_scalar("losses/entropy_lo", log_dict["lo_entropy_loss"], log_dict["global_step"])
            self.writer.add_scalar("losses/entropy_hi", log_dict["hi_entropy_loss"], log_dict["global_step"])

            self.writer.add_scalar("losses/approx_kl_z", log_dict["approx_kl_z"], log_dict["global_step"])
            self.writer.add_scalar("losses/clipfrac_z", np.mean(log_dict["clipfracs_z"]), log_dict["global_step"])
            self.writer.add_scalar("losses/explained_variance_z", log_dict["explained_var_z"], log_dict["global_step"])
            self.writer.add_scalar("losses/explained_variance", log_dict["explained_var"], log_dict["global_step"])
        
        elif self.config.model.name == "fcp":
            self.writer.add_scalar("losses/value_loss", log_dict["v_loss"], log_dict["global_step"])
            self.writer.add_scalar("losses/policy_loss", log_dict["pg_loss"], log_dict["global_step"])
            self.writer.add_scalar("losses/entropy", log_dict["entropy_loss"], log_dict["global_step"])
            self.writer.add_scalar("losses/approx_kl", log_dict["approx_kl"], log_dict["global_step"])
            self.writer.add_scalar("losses/clipfrac", np.mean(log_dict["clipfracs"]), log_dict["global_step"])

    def log_train_pop_info(self, log_dict):
        self.writer.add_scalar("charts/SPS", int(log_dict["global_step"] / (time.time() - log_dict["start_time"])), log_dict["global_step"])
        self.writer.add_scalar("charts/learning_rate", log_dict["lr"], log_dict["global_step"])
        self.writer.add_scalar("losses/jsd_loss", log_dict["jsd"], log_dict["global_step"])
        self.writer.add_scalar("losses/average_entopy", log_dict["avg_entropy"], log_dict["global_step"])
        self.writer.add_scalar("losses/entropy_of_average_policy", log_dict["entropy_of avg"], log_dict["global_step"])

        for agent_ind in range(self.config.training.population_size):
            average_reward = np.mean([info["episode"]["r"] for info in log_dict["infos"][agent_ind]])
            self.writer.add_scalar("charts/avg_episodic_return_agent_{}".format(agent_ind), average_reward, log_dict["global_step"])
            # self.writer.add_scalar("losses/value_loss_agent_{}".format(agent_ind), log_dict["pg_losses"][agent_ind], log_dict["global_step"])
            # self.writer.add_scalar("losses/policy_loss_agent_{}".format(agent_ind), log_dict["v_losses"][agent_ind], log_dict["global_step"])
            self.writer.add_scalar("losses/entropy_agent_{}".format(agent_ind), log_dict["entropy_losses"][agent_ind], log_dict["global_step"])
