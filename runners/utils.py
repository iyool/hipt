import numpy as np
import torch

def compute_advantage(episode_traj, config):
    with torch.no_grad():
        lastgaelam = 0
        for t in reversed(range(config.training.rollout_steps)):
            if t == config.training.rollout_steps - 1:
                nextnonterminal = 0
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - episode_traj["dones"][t + 1]
                nextvalues = episode_traj["values"][t + 1]

            delta = episode_traj["rewards"][t] + config.training.gamma * nextvalues * nextnonterminal - episode_traj["values"][t]
            episode_traj["advantages"][t] = lastgaelam = delta + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam
        episode_traj["returns"] = episode_traj["advantages"] + episode_traj["values"]
    return episode_traj

def compute_advantage_pop(episode_traj, config):
    with torch.no_grad():
        for agent_ind in range(config.training.population_size):
            lastgaelam_p0 = 0
            lastgaelam_p1 = 0
            for t in reversed(range(config.training.rollout_steps)):
                if t == config.training.rollout_steps - 1:
                    nextnonterminal = 0
                    nextvalues_p0 = 0
                    nextvalues_p1 = 0
                else:
                    nextnonterminal = 1.0 - episode_traj["dones"][agent_ind,t + 1]
                    nextvalues_p0 = episode_traj["p0"]["values"][agent_ind,t + 1]
                    nextvalues_p1 = episode_traj["p1"]["values"][agent_ind,t + 1]
                delta_p0 = episode_traj["rewards"][agent_ind, t] + config.training.gamma * nextvalues_p0 * nextnonterminal - episode_traj["p0"]["values"][agent_ind, t]
                episode_traj["p0"]["advantages"][agent_ind, t] = lastgaelam_p0 = delta_p0 + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam_p0
                delta_p1 = episode_traj["rewards"][agent_ind, t] + config.training.gamma * nextvalues_p1 * nextnonterminal - episode_traj["p1"]["values"][agent_ind, t]
                episode_traj["p1"]["advantages"][agent_ind, t] = lastgaelam_p1 = delta_p1 + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam_p1
        episode_traj["p0"]["returns"] = episode_traj["p0"]["advantages"] + episode_traj["p0"]["values"]
        episode_traj["p1"]["returns"] = episode_traj["p1"]["advantages"] + episode_traj["p1"]["values"]
    return episode_traj

def compute_advantage_hi(episode_traj, config):
    with torch.no_grad():
        for ind in range(config.training.num_envs):
            lastgaelam_z = 0
            for t in reversed(range(episode_traj[ind]["hi_rewards"].shape[0])):
                if t == episode_traj[ind]["hi_rewards"].shape[0] - 1:
                    nextnonterminal = 0.0
                    nextvalues_z = 0.0
                else:
                    nextnonterminal = 1.0
                    nextvalues_z = episode_traj[ind]["z_values"][t + 1]
                delta_z = episode_traj[ind]["hi_rewards"][t] + config.training.gamma * nextvalues_z * nextnonterminal - episode_traj[ind]["z_values"][t]
                episode_traj[ind]["z_adv"][t] = lastgaelam_z = delta_z + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam_z
            episode_traj[ind]["z_ret"] = episode_traj[ind]["z_adv"] + episode_traj[ind]["z_values"]
    return episode_traj

