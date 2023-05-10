import os
import gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from trainers.hipt_trainer import HiPTTrainer
from utils.utils import make_env
from tqdm import tqdm


layout_name_map = {
        "cramped_room": "Cramped Room",
        "asymmetric_advantages": "Asymmetric Advantages",
        "counter_circuit_o_1order": "Counter Circuit",
        "forced_coordination": "Forced Coordination",
        "coordination_ring": "Coordination Ring",
    }

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif',fps = 60):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=fps)
    anim.save(path + filename, writer='imagemagick', fps=fps)

    plt.close()

def eval_rollout(trainer, config):
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.gym_id, config.layout.name, config.seed + config.training.num_envs + i) for i in range(config.training.population_size*2)]
    )
    initial_obs = envs.reset()

    cum_rew = np.zeros(config.training.population_size*2)
    if isinstance(trainer, HiPTTrainer):
        curr_p_len = np.random.randint(config.training.p_range[0], config.training.p_range[1], size = config.training.population_size*2)

    next_obs_p0 = torch.Tensor(initial_obs[:,0]).to(trainer.device)
    next_obs_p1 = torch.Tensor(initial_obs[:,1]).to(trainer.device)

    lstm_state = (
        torch.zeros(trainer.agent.lstm.num_layers, config.training.population_size*2, trainer.agent.lstm.hidden_size).to(trainer.device),
        torch.zeros(trainer.agent.lstm.num_layers, config.training.population_size*2, trainer.agent.lstm.hidden_size).to(trainer.device),
    )   

    if config.evaluation.visualize_ep:
        visualizer = StateVisualizer()
        vis_path = os.path.join(os.getcwd(), "visualizations")
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)
        ep_visualizations = [[] for _ in range(config.training.population_size*2)]

    next_done = torch.zeros(2*config.training.population_size).to(trainer.device)

    print("Evaluating agent with SP Population.....")

    for step in range(config.training.rollout_steps):
        current_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())
        agent_obs = torch.cat((next_obs_p0[:config.training.population_size], next_obs_p1[config.training.population_size:]), dim = 0)
        
        if config.evaluation.visualize_ep:
            for ind,env in enumerate(envs.envs):
                hud_data = {"Timestep" : step, "Reward" : cum_rew[ind]}
                surface = visualizer.render_state(env.base_env.state, grid = env.mdp.terrain_mtx, hud_data = hud_data)
                state_imgdata = pygame.surfarray.array3d(surface).swapaxes(0,1)
                ep_visualizations[ind].append(state_imgdata)

        if isinstance(trainer, HiPTTrainer):
            z, _, __, ___, lstm_state, ____ = trainer.agent.get_z_and_value(agent_obs, next_done, lstm_state)
            if step == 0:
                current_z = z.clone()
            else:
                for p_step in range(config.training.population_size*2):
                    if curr_p_len[p_step] == 0:
                        curr_p_len[p_step] = np.random.randint(config.training.p_range[0], config.training.p_range[1])
                        current_z[p_step] = z[p_step]
            agent_actions, _, __, ___, _____, ______ = trainer.agent.get_action_and_value(agent_obs, current_z, next_done, current_lstm_state)

        else:
            agent_actions, _, __, ___, lstm_state = trainer.agent.get_action_and_value(agent_obs, next_done, current_lstm_state)

        partner_actions = torch.zeros(config.training.population_size*2).to(trainer.device)
        for ind, partner in enumerate(trainer.train_partners):
            partner_obs = torch.cat((next_obs_p1[ind:ind+1], next_obs_p0[ind+config.training.population_size:ind+config.training.population_size + 1]), dim = 0)
            partner_action,__, _, ___, ____ = partner.get_action_and_value(partner_obs)
            partner_actions[ind] = partner_action[0]
            partner_actions[ind+config.training.population_size] = partner_action[1]    

        actions_p0 = torch.cat((agent_actions[:config.training.population_size], partner_actions[:config.training.population_size]), dim = 0)
        actions_p1 = torch.cat((partner_actions[:config.training.population_size], agent_actions[:config.training.population_size]), dim = 0)
        joint_action = torch.cat((actions_p0.view(2*config.training.population_size,1),actions_p1.view(2*config.training.population_size,1)),1)
        joint_action = joint_action.type(torch.int8)
        next_obs, reward, done, info = envs.step(joint_action.cpu().numpy())

        next_obs_p0 = torch.Tensor(next_obs[:,0]).to(trainer.device)
        next_obs_p1 = torch.Tensor(next_obs[:,1]).to(trainer.device)
        next_done = torch.Tensor(done).to(trainer.device)

        cum_rew += reward

        if isinstance(trainer, HiPTTrainer):
            curr_p_len -= 1
    
    envs.close()

    for ind, reward in enumerate(cum_rew):
        if ind < config.training.population_size:
            print(f"Blue : {config.model.name} Agent, Green : SP Agent {ind}, Total reward: {reward}")
            if config.evaluation.visualize_ep:
                save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_{config.model.name}_green_spagent{ind}.gif")   
        else:
            print(f"Blue : SP Agent {ind%config.training.population_size}, Green : {config.model.name} Agent, Total reward: {reward}")
            if config.evaluation.visualize_ep:
                save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_spagent{ind%config.training.population_size}_green_{config.model.name}.gif")

def plot_sp_heatmap(trainer, config):
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.gym_id, config.layout.name, config.seed + config.training.num_envs + i) for i in range(config.training.population_size)]
    )
    initial_obs = envs.reset()

    if config.layout.eval_partner_pop_path is not None:
        trainer.load_population(config.evaluation.agent_type,config.layout.eval_partner_pop_path)

    vis_path = os.path.join(os.getcwd(), "visualizations")
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    rew_mat = np.zeros((config.training.population_size, config.training.population_size))

    headers = [f"Agent {i}" for i in range(config.training.population_size)]

    rows = [[f"Agent {i}"] for i in range(config.training.population_size)]

    next_obs_p0 = torch.Tensor(initial_obs[:,0]).to(trainer.device)
    next_obs_p1 = torch.Tensor(initial_obs[:,1]).to(trainer.device)

    for agent_ind in range(config.training.population_size):
        print(f"Running rollouts for Agent {agent_ind}.....")
        for step in tqdm(range(config.training.rollout_steps)):
            agent_actions,__, _, ___, ____ = trainer.agent_pop[agent_ind].get_action_and_value(next_obs_p0)
            partner_actions = torch.zeros(config.training.population_size).to(trainer.device)
            for partner_ind in range(config.training.population_size):                 
                partner_action,__, _, ___, ____ = trainer.agent_pop[partner_ind].get_action_and_value(next_obs_p1[partner_ind:partner_ind+1])
                partner_actions[partner_ind] = partner_action[0]

            joint_action = torch.cat((agent_actions.view(config.training.population_size,1),partner_actions.view(config.training.population_size,1)),1)
            joint_action = joint_action.type(torch.int8)
            next_obs, reward, done, info = envs.step(joint_action.cpu().numpy())

            rew_mat[agent_ind] += reward

            next_obs_p0 = torch.Tensor(next_obs[:,0]).to(trainer.device)
            next_obs_p1 = torch.Tensor(next_obs[:,1]).to(trainer.device)
        rows[agent_ind].extend([str(rew) for rew in rew_mat[agent_ind]])
    
    print("Plotting Heatmap.....")

    plt.rcParams["figure.figsize"] = (20,15)
    labels = [rows[i][1:] for i in range(config.training.population_size)]
    
    ax = sns.heatmap(rew_mat, cmap="inferno", annot=labels, annot_kws={'fontsize': 8}, fmt='s', xticklabels = headers, yticklabels = headers)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    plt.title(layout_name_map[f"{config.layout.name}"])
    plt.savefig(vis_path + f"/sp_{config.layout.name}_heatmap_.png",bbox_inches='tight')
    plt.close()

    print("Done!")
    
