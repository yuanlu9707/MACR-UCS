from algorithms.algo.agent.DPPO import DPPOAgent
import torch.nn as nn
from algorithms.models import CategoricalActor
from torch.optim import Adam
from algorithms.models import MLP
from algorithms.crgc import CRGCActor

class IPPOAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        DPPOAgent.__init__(self, logger, device, agent_args, input_args)

        self.actors = self._init_actors()
        self.vs = self._init_vs()
        self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)
        self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr_v)

    def _init_actors(self):
        if self.input_args.use_crgc:
            actor = CRGCActor(
                n_uav=self.n_agent,
                n_user=self.input_args.poi_num,
                frame_obs_dim=self.crgc_frame_observation_dim,
                max_episode_step=self.input_args.max_episode_step,
                branch_sizes=self.pi_args.branchs,
                hidden_dim=int(self.pi_args.sizes[-1]),
                args=self.input_args,
            ).to(self.device)
            return nn.ModuleList([actor])
        actors = nn.ModuleList()
        for i in range(self.n_agent):
            self.pi_args.sizes[0] = self.observation_dim
            if self.input_args.obs_add_action:
                self.pi_args.sizes[0] += 2
            actors.append(CategoricalActor(**self.pi_args._toDict()).to(self.device))
        return actors

    def _init_vs(self):
        vs = nn.ModuleList()
        for i in range(self.n_agent):
            self.v_args.sizes[0] = self.observation_dim
            if self.input_args.obs_add_action and not self.input_args.use_crgc:
                self.v_args.sizes[0] += 2
            if self.input_args.MAPPO:
                self.v_args.sizes[0] *= self.n_agent
            vs.append(MLP(**self.v_args._toDict()).to(self.device))
        return vs
