from cs285.models.ff_model import FFModel
from .base_agent import BaseAgent
from .sac_agent import SACAgent
from .mb_agent import MBAgent
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *

class MBPOAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBPOAgent, self).__init__()
        self.mb_agent = MBAgent(env, agent_params)
        self.sac_agent = SACAgent(env, agent_params['sac_params'])
        self.env = env

        self.actor = self.sac_agent.actor

    def train(self, *args):
        return self.mb_agent.train(*args)
    
    def train_sac(self, *args):
        return self.sac_agent.train(*args)

    def collect_model_trajectory(self, rollout_length=1):
        # TODO (Q6): Collect a trajectory of rollout_length from the learned 
        # dynamics model. Start from a state sampled from the replay buffer.

        # sample 1 transition from self.mb_agent.replay_buffer
        # print('collect_model_trajectory')
        ob, _, _, _, terminal = self.mb_agent.replay_buffer.sample_random_data(1)

        obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
        for _ in range(rollout_length):
            # get the action from the policy
            # print(f'rollout_length: {_}/{rollout_length}')
            ac = self.actor.get_action(ob, True)
            
            # determine the next observation by averaging the prediction of all the 
            # dynamics models in the ensemble
            next_ob = 0.0
            for i in range(self.mb_agent.ensemble_size):
                model: FFModel = self.mb_agent.dyn_models[i]
                next_ob += model.get_prediction(ob, ac, self.mb_agent.data_statistics)
            next_ob /= self.mb_agent.ensemble_size

            # query the reward function to determine the reward of this transition
            # HINT: use self.env.get_reward
            rew, _ = self.env.get_reward(next_ob, ac)

            obs.append(ob[0])
            acs.append(ac[0])
            rewards.append(rew[0])
            next_obs.append(next_ob[0])
            terminals.append(terminal[0])

            ob = next_ob
        # print('collect_model_trajectory done')
        return [Path(obs, image_obs, acs, rewards, next_obs, terminals)]

    def add_to_replay_buffer(self, paths, from_model=False, **kwargs):
        self.sac_agent.add_to_replay_buffer(paths)
        # only add rollouts from the real environment to the model training buffer
        if not from_model:
            self.mb_agent.add_to_replay_buffer(paths, **kwargs)

    def sample(self, *args, **kwargs):
        return self.mb_agent.sample(*args, **kwargs)

    def sample_sac(self, *args, **kwargs):
        return self.sac_agent.sample(*args, **kwargs)