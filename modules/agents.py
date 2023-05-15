import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np
from utils.utils import layer_init


class CNNAgent(nn.Module):
    def __init__(self, config):
        super(CNNAgent, self).__init__()
        self.no_filters = config.layout.observation_shape[0]
        self.feature_dim = np.prod(config.layout.observation_shape)
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self.no_filters, self.no_filters,5,padding = 'same')),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(self.no_filters, self.no_filters,3,padding = 'same')),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(self.no_filters, self.no_filters,3,padding = 'same')),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(self.feature_dim, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
        )
        self.actor = layer_init(nn.Linear(512, config.layout.action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), probs.probs

class LSTMAgent(nn.Module):
    def __init__(self, config):
        super(LSTMAgent, self).__init__()
        self.no_filters = config.layout.observation_shape[0]
        self.feature_dim = np.prod(config.layout.observation_shape)
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self.no_filters, self.no_filters,5,padding = 'same')),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(self.no_filters, self.no_filters,3,padding = 'same')),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(self.no_filters, self.no_filters,3,padding = 'same')),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(self.feature_dim, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
        )

        self.lstm = nn.LSTM(512, 512)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(nn.Linear(512, config.layout.action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x, done, lstm_state):
        feat, lstm_state = self.get_states(x, lstm_state, done)
        return self.critic(feat)

    def get_action_and_value(self, x, done, lstm_state, action=None):
        feat, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(feat)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(feat), lstm_state

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)
        batch_size = lstm_state[0].shape[1]

        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            nh, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [nh]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state


class HiPPOAgent(nn.Module):
    def __init__(self, config):
        super(HiPPOAgent, self).__init__()
        self.no_filters = config.layout.observation_shape[0]
        self.z_dim = config.layout.z_dim
        self.feature_dim = np.prod(config.layout.observation_shape)
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self.no_filters, self.no_filters,5,padding = 'same')),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(self.no_filters, self.no_filters,3,padding = 'same')),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(self.no_filters, self.no_filters,3,padding = 'same')),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(self.feature_dim, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),

        )

        self.lstm = nn.LSTM(512, 512)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)


        self.hi_feat = nn.Sequential(layer_init(nn.Linear(512, 128)), nn.LeakyReLU())
        self.hi_actor = layer_init(nn.Linear(128, self.z_dim), std=0.01)
        self.hi_critic = layer_init(nn.Linear(128, 1), std=1)

        self.lo_feat = nn.Sequential(layer_init(nn.Linear(512 + self.z_dim, 128)), nn.LeakyReLU())
        self.lo_actor = layer_init(nn.Linear(128, config.layout.action_dim), std=0.01)
        self.lo_critic = layer_init(nn.Linear(128, 1), std=1)

        self.disc_feat = nn.Sequential(layer_init(nn.Linear(512, 128)), nn.LeakyReLU())
        self.discriminator = layer_init(nn.Linear(128, self.z_dim), std=0.01)

    def get_states(self, x, lstm_state, done, env_first):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        if env_first:
            hidden = torch.transpose(hidden.reshape((batch_size, -1, self.lstm.input_size)),0,1)
            done = torch.transpose(done.reshape((batch_size,-1)),0,1)
        else:
            hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
            done = done.reshape((-1, batch_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            nh, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [nh]
        if env_first:
            new_hidden = torch.flatten(torch.transpose(torch.cat(new_hidden), 0, 1), 0, 1)
        else:
            new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state


    def get_hivalue(self, x, done, lstm_state, t_ind = None, env_first = False):
        feat, lstm_state = self.get_states(x, lstm_state, done, env_first)
        if t_ind is not None:
            feat = feat[t_ind]
        hi_feat = self.high_feat(feat)
        return self.hi_critic(hi_feat)

    def get_lovalue(self, x, z, done, lstm_state, env_first = False):
        feat, lstm_state = self.get_states(x, lstm_state, done, env_first)
        z = F.one_hot(z, num_classes = self.z_dim)
        lo_feat = torch.cat((feat,z),-1)
        lo_feat = self.lo_feat(lo_feat)
        return self.lo_critic(lo_feat)

    def get_action_and_value(self, x, z, done, lstm_state, action=None, env_first=False, expanded=False):
        feat, lstm_state = self.get_states(x, lstm_state, done, env_first)
        z = F.one_hot(z, num_classes = self.z_dim)
        if expanded:
            feat = torch.tile(feat, (self.z_dim, 1))
        lo_feat = torch.cat((feat,z),-1)
        lo_feat = self.lo_feat(lo_feat)
        logits = self.lo_actor(lo_feat)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.lo_critic(lo_feat), lstm_state, probs.probs

    def get_z_and_value(self, x, done, lstm_state, z = None, t_ind = None, env_first = False):
        feat, lstm_state = self.get_states(x, lstm_state, done, env_first)
        if t_ind is not None:
            feat = feat[t_ind]
        hi_feat = self.hi_feat(feat)
        logits = self.hi_actor(hi_feat)
        probs = Categorical(logits=logits)
        if z is None:
            z = probs.sample()
        return z, probs.log_prob(z), probs.entropy(), self.hi_critic(hi_feat), lstm_state , probs.probs