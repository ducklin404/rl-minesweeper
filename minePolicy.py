import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class ConvTrunk(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 64):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )


    def forward(self, observation):
        output = self.cnn(observation)
        return output


class ConvHeads(nn.Module):
    def __init__(self, channels, n_kinds, n_actions, vf_dim=64):
        super().__init__()
        self.latent_dim_pi = n_actions
        self.latent_dim_vf = vf_dim

        self.actor = nn.Conv2d(channels, n_kinds, kernel_size=1)

        self.critic = nn.Sequential(
            nn.Linear(channels, vf_dim),
            nn.ReLU(),
        )

    def forward_actor(self, features):
        logits = self.actor(features)
        return logits.flatten(1)

    def forward_critic(self, features):
        pooled = features.mean(dim=(2, 3))
        return self.critic(pooled)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class MinePolicy(MaskableActorCriticPolicy):

    def _build_mlp_extractor(self):
        _, h, w = self.observation_space.shape
        n_actions = self.action_space.n
        n_kinds = n_actions // (h * w)
        self.mlp_extractor = ConvHeads(self.features_dim, n_kinds, n_actions)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)

        self.action_net = nn.Identity()

        if self.ortho_init:
            self.mlp_extractor.actor.apply(lambda m: self.init_weights(m, gain=0.01))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


if __name__ == '__main__':
    from minesweeper_env import MineEnv
    space = MineEnv().observation_space

    net = ConvTrunk(space)
    feats = net(torch.zeros(1, *space.shape))
    print("trunk:", feats.shape, net(torch.zeros(1, 11, 16, 30)).shape)

    heads = ConvHeads(channels=64, n_kinds=3, n_actions=243)
    print("actor:", heads.forward_actor(feats).shape)
    print("critic:", heads.forward_critic(feats).shape)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    env = ActionMasker(MineEnv(), lambda env: env.action_masks())
    model = MaskablePPO(MinePolicy, env,
                        policy_kwargs={"features_extractor_class": ConvTrunk, "normalize_images": False})
    print(model.policy)

    obs, _ = env.reset(seed=0)
    obs_t = torch.as_tensor(obs[None])
    mask = env.action_masks()
    dist = model.policy.get_distribution(obs_t, action_masks=mask[None])
    probs = dist.distribution.probs[0].detach().numpy()
    print("probs:", probs.shape, "on legal cells:", probs[mask].sum())
    print("value:", model.policy.predict_values(obs_t).shape)

    model.learn(2048)
    print("all fuking good")
