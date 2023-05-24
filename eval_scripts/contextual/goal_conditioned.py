import numpy as np
import gym

from eval_scripts.utils.io import load_local_or_remote_file
from eval_scripts.distribution import DictDistribution


class PresampledPathDistribution(DictDistribution):
    def __init__(
            self,
            datapath,
            representation_size,
            # Set to true if you plan to re-encode presampled images
            initialize_encodings=True,
    ):
        self._presampled_goals = load_local_or_remote_file(datapath)
        self.representation_size = representation_size
        self._num_presampled_goals = self._presampled_goals[list(
            self._presampled_goals)[0]].shape[0]

        if initialize_encodings:
            self._presampled_goals['initial_latent_state'] = np.zeros(
                (self._num_presampled_goals, self.representation_size))

        self._set_spaces()

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._num_presampled_goals, batch_size)
        sampled_goals = {
            k: v[idx] for k, v in self._presampled_goals.items()
        }
        return sampled_goals

    def _set_spaces(self):
        pairs = []
        for key in self._presampled_goals:
            dim = self._presampled_goals[key][0].shape[0]
            box = gym.spaces.Box(-np.ones(dim), np.ones(dim))
            pairs.append((key, box))
        self.observation_space = gym.spaces.Dict(pairs)

    @property
    def spaces(self):
        return self.observation_space.spaces
