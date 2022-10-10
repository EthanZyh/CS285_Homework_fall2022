from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = ptu.from_numpy(observation)
        with torch.no_grad():
            if sample:
                action = self.forward(observation).rsample()
            else:
                action = self.forward(observation).mean
        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        lb, ub = self.action_range
        logstd = lb + .5 * (ub - lb) * (self.logstd.tanh() + 1) # (-1,1) -> (lb, ub)
        std = torch.exp(logstd)
        mean = self.mean_net.forward(observation)
        SN = sac_utils.SquashedNormal(mean, std)
        return SN

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        
        # calc entropy
        '''  [The next two lines of comments are wrong!!! (why?)]'''
        '''action = ptu.from_numpy(self.get_action(obs, True))'''
        '''action_dist = self.forward(ptu.from_numpy(obs))'''
        action_dist = self.forward(ptu.from_numpy(obs))
        action = action_dist.rsample().clip(*self.action_range)
        log_prob = action_dist.log_prob(action)
        entropy = -torch.sum(log_prob, dim=1)

        # calc actor_loss & alpha_loss
        q1, q2 = critic.forward(ptu.from_numpy(obs), action)
        qmin = torch.min(q1, q2)
        actor_loss = torch.mean(-qmin - self.alpha.detach() * entropy)
        entropy_term_detach = (self.target_entropy - entropy).detach() # independent of alpha!!!
        alpha_loss = -torch.mean(self.alpha * entropy_term_detach)

        # optimize
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha