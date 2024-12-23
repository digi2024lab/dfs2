import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from preprocess import Preprocessor  # Import the updated Preprocessor
from replay_buffer import Batch

def get_net(
        num_in: int,
        num_out: int,
        final_activation,
        num_hidden_layers: int = 6,
        num_neurons_per_hidden_layer: int = 256
    ) -> nn.Sequential:

    layers = []

    layers.extend([
        nn.Linear(num_in, num_neurons_per_hidden_layer),
        nn.SELU(),
    ])

    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer),
            nn.SELU(),
        ])

    layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)

class NormalPolicyNet(nn.Module):
    def __init__(self, feature_dim, scalar_dim, action_dim):
        super(NormalPolicyNet, self).__init__()
        self.input_dim = feature_dim + scalar_dim
        self.shared_net = get_net(
            num_in=self.input_dim,
            num_out=256,
            final_activation=nn.SELU()
        )
        self.output_layer = nn.Linear(256, 2 * action_dim)  # Outputs concatenated means and log_stds

    def forward(self, features: torch.tensor, scalars: torch.tensor):
        combined_input = torch.cat([features, scalars], dim=1)
        out = self.shared_net(combined_input)
        out = self.output_layer(out)  # Single tensor output
        return out  # Shape: (batch_size, 2 * action_dim)

class QNet(nn.Module):
    def __init__(self, feature_dim, scalar_dim, action_dim):
        super(QNet, self).__init__()
        self.input_dim = feature_dim + scalar_dim + action_dim
        self.net = get_net(num_in=self.input_dim, num_out=1, final_activation=None)

    def forward(self, features: torch.tensor, scalars: torch.tensor, actions: torch.tensor):
        combined_input = torch.cat([features, scalars, actions], dim=1)
        return self.net(combined_input)

class ParamsPool:
    def __init__(self, feature_dim, scalar_dim, action_dim, input_height, input_width, activate_scale=False, device=None):
        self.device = device if device is not None else torch.device('cpu')
        self.action_limit_low = -1.0
        self.action_limit_high = 1.0
        self.activate_scale = activate_scale

        self.feature_dim = feature_dim
        self.scalar_dim = scalar_dim
        self.action_dim = action_dim

        # Initialize Preprocessor
        self.preprocessor = Preprocessor(
            input_channels=1,
            output_size=feature_dim,
            input_height=input_height,
            input_width=input_width
        ).to(self.device)

        # Initialize Policy Network
        self.Normal = NormalPolicyNet(
            feature_dim=self.feature_dim,
            scalar_dim=self.scalar_dim,
            action_dim=action_dim
        ).to(self.device)

        # Initialize Q-Networks
        self.Q1 = QNet(
            feature_dim=self.feature_dim,
            scalar_dim=self.scalar_dim,
            action_dim=action_dim
        ).to(self.device)

        self.Q1_targ = QNet(
            feature_dim=self.feature_dim,
            scalar_dim=self.scalar_dim,
            action_dim=action_dim
        ).to(self.device)
        self.Q1_targ.load_state_dict(self.Q1.state_dict())

        self.Q2 = QNet(
            feature_dim=self.feature_dim,
            scalar_dim=self.scalar_dim,
            action_dim=action_dim
        ).to(self.device)

        self.Q2_targ = QNet(
            feature_dim=self.feature_dim,
            scalar_dim=self.scalar_dim,
            action_dim=action_dim
        ).to(self.device)
        self.Q2_targ.load_state_dict(self.Q2.state_dict())

        # Hyperparameters
        self.gamma = 0.99999
        self.polyak = 0.995

        # Entropy Parameter alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(0.1)).to(self.device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)

        # Optimizers
        self.preprocessor_optimizer = optim.Adam(self.preprocessor.parameters(), lr=1e-4)
        self.Normal_optimizer = optim.Adam(self.Normal.parameters(), lr=1e-4)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=1e-4)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=1e-4)

        # DataParallel wrapping (only when necessary)
        self.wrap_models_with_dataparallel()

    def wrap_models_with_dataparallel(self):
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")

            # Wrap models with DataParallel for multi-GPU training
            self.Normal = nn.DataParallel(self.Normal)
            self.Q1 = nn.DataParallel(self.Q1)
            self.Q2 = nn.DataParallel(self.Q2)
            self.preprocessor = nn.DataParallel(self.preprocessor)

            self.Normal.to(self.device)
            self.Q1.to(self.device)
            self.Q2.to(self.device)
            self.preprocessor.to(self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def min_i_12(self, a: torch.tensor, b: torch.tensor) -> torch.tensor:
        return torch.min(a, b)

    def sample_action_and_compute_log_pi(self, features: torch.tensor, scalars: torch.tensor, use_reparametrization_trick: bool) -> tuple:
        outputs = self.Normal(features, scalars)
        if isinstance(outputs, (tuple, list)):
            outputs = torch.cat(outputs, dim=0)
        # Split the outputs into means and log_stds
        means, log_stds = torch.chunk(outputs, 2, dim=-1)

        LOG_STD_MAX = 2
        LOG_STD_MIN = -20
        log_stds = torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        stds = torch.exp(log_stds)

        mu_given_s = Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)
        u = mu_given_s.rsample() if use_reparametrization_trick else mu_given_s.sample()
        a = torch.tanh(u)
        log_pi_a_given_s = mu_given_s.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1)
        return a, log_pi_a_given_s

    # Gradient clipping to prevent exploding gradients
    def clip_gradient(self, net: nn.Module) -> None:
        for param in net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

    # Polyak update (soft update)
    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    def scale_action(self, action):
        min_action = self.action_limit_low
        max_action = self.action_limit_high
        scaled_action = min_action + (action + 1.0) * 0.5 * (max_action - min_action)
        return scaled_action

    # Methods for learning
    def update_networks(self, b: Batch) -> None:
        # Wrap models with DataParallel if not already wrapped
        if not isinstance(self.preprocessor, nn.DataParallel) and torch.cuda.device_count() > 1:
            self.wrap_models_with_dataparallel()

        # Get features for current state
        features_s, x_recon_s = self.preprocessor(b.img)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(x_recon_s, b.img)

        # Step 12: calculating targets
        with torch.no_grad():
            features_ns = self.preprocessor(b.n_img, reconstruct=False)
            # Calculate next action using next state
            na, log_pi_na_given_ns = self.sample_action_and_compute_log_pi(
                features_ns, b.n_scalars, use_reparametrization_trick=False)
            # Calculate next q values
            q1_next = self.Q1_targ(features_ns, b.n_scalars, na)
            q2_next = self.Q2_targ(features_ns, b.n_scalars, na)
            min_q_next = self.min_i_12(q1_next, q2_next)
            # Calculate the target values
            targets = b.r + self.gamma * (1 - b.d) * (
                min_q_next - self.alpha.detach() * log_pi_na_given_ns.unsqueeze(-1))

        # Detach features when updating Q-networks
        features_s_detached = features_s.detach()

        # Step 13: learning the Q functions
        Q1_predictions = self.Q1(features_s_detached, b.scalars, b.a)
        Q1_loss = F.mse_loss(Q1_predictions, targets)
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.clip_gradient(net=self.Q1)
        self.Q1_optimizer.step()

        Q2_predictions = self.Q2(features_s_detached, b.scalars, b.a)
        Q2_loss = F.mse_loss(Q2_predictions, targets)
        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.clip_gradient(net=self.Q2)
        self.Q2_optimizer.step()

        # Step 14: learning the policy
        a, log_pi_a_given_s = self.sample_action_and_compute_log_pi(
            features_s, b.scalars, use_reparametrization_trick=True)
        q1_pi = self.Q1(features_s, b.scalars, a)
        q2_pi = self.Q2(features_s, b.scalars, a)
        min_q_pi = self.min_i_12(q1_pi, q2_pi)
        policy_loss = (self.alpha.detach() * log_pi_a_given_s - min_q_pi).mean() + 0.1 * recon_loss  # Add recon loss with weight

        self.Normal_optimizer.zero_grad()
        self.preprocessor_optimizer.zero_grad()  # Zero the gradients of the preprocessor
        policy_loss.backward()
        self.clip_gradient(net=self.Normal)
        self.clip_gradient(net=self.preprocessor)
        self.Normal_optimizer.step()
        self.preprocessor_optimizer.step()  # Update the preprocessor

        # Step 15: update temperature
        alpha_loss = -(self.log_alpha * (log_pi_a_given_s + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Polyak update for the q networks
        with torch.no_grad():
            self.polyak_update(old_net=self.Q1_targ, new_net=self.Q1)
            self.polyak_update(old_net=self.Q2_targ, new_net=self.Q2)

    # Method for the agent to act in the environment
    def act(self, image: np.array, scalars: np.array) -> np.array:
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(self.device)  # Shape: (1, 1, H, W)
        scalars_tensor = torch.tensor(scalars).unsqueeze(0).float().to(self.device)  # Shape: (1, scalar_dim)

        # Use original models (not DataParallel) during inference
        preprocessor = self.preprocessor.module if isinstance(self.preprocessor, nn.DataParallel) else self.preprocessor
        normal_policy_net = self.Normal.module if isinstance(self.Normal, nn.DataParallel) else self.Normal

        # Get features from preprocessor
        with torch.no_grad():
            features = preprocessor(image_tensor, reconstruct=False)

        outputs = normal_policy_net(features, scalars_tensor)
        if isinstance(outputs, (tuple, list)):
            outputs = torch.cat(outputs, dim=0)
        means, log_stds = torch.chunk(outputs, 2, dim=-1)

        LOG_STD_MAX = 2
        LOG_STD_MIN = -20
        log_stds = torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        stds = torch.exp(log_stds)

        mu_given_s = Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)
        action = mu_given_s.sample()
        action = torch.tanh(action)

        action = action.cpu().numpy()[0]
        if self.activate_scale:
            scaled_action = self.scale_action(action)
            return scaled_action
        else:
            return action

    # Save model parameters
    def save_model(self, save_path: str) -> None:
        # Save the state_dict of the original models
        preprocessor = self.preprocessor.module if isinstance(self.preprocessor, nn.DataParallel) else self.preprocessor
        normal_policy_net = self.Normal.module if isinstance(self.Normal, nn.DataParallel) else self.Normal
        Q1 = self.Q1.module if isinstance(self.Q1, nn.DataParallel) else self.Q1
        Q2 = self.Q2.module if isinstance(self.Q2, nn.DataParallel) else self.Q2
        Q1_targ = self.Q1_targ.module if isinstance(self.Q1_targ, nn.DataParallel) else self.Q1_targ
        Q2_targ = self.Q2_targ.module if isinstance(self.Q2_targ, nn.DataParallel) else self.Q2_targ

        torch.save({
            'preprocessor_state_dict': preprocessor.state_dict(),
            'Normal_state_dict': normal_policy_net.state_dict(),
            'Q1_state_dict': Q1.state_dict(),
            'Q2_state_dict': Q2.state_dict(),
            'Q1_targ_state_dict': Q1_targ.state_dict(),
            'Q2_targ_state_dict': Q2_targ.state_dict(),
            'preprocessor_optimizer_state_dict': self.preprocessor_optimizer.state_dict(),
            'Normal_optimizer_state_dict': self.Normal_optimizer.state_dict(),
            'Q1_optimizer_state_dict': self.Q1_optimizer.state_dict(),
            'Q2_optimizer_state_dict': self.Q2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }, save_path)

    def load_model(self, load_path: str) -> None:
        checkpoint = torch.load(load_path, map_location=self.device)

        preprocessor = self.preprocessor.module if isinstance(self.preprocessor, nn.DataParallel) else self.preprocessor
        normal_policy_net = self.Normal.module if isinstance(self.Normal, nn.DataParallel) else self.Normal
        Q1 = self.Q1.module if isinstance(self.Q1, nn.DataParallel) else self.Q1
        Q2 = self.Q2.module if isinstance(self.Q2, nn.DataParallel) else self.Q2
        Q1_targ = self.Q1_targ.module if isinstance(self.Q1_targ, nn.DataParallel) else self.Q1_targ
        Q2_targ = self.Q2_targ.module if isinstance(self.Q2_targ, nn.DataParallel) else self.Q2_targ

        preprocessor.load_state_dict(checkpoint['preprocessor_state_dict'])
        normal_policy_net.load_state_dict(checkpoint['Normal_state_dict'])
        Q1.load_state_dict(checkpoint['Q1_state_dict'])
        Q2.load_state_dict(checkpoint['Q2_state_dict'])
        Q1_targ.load_state_dict(checkpoint['Q1_targ_state_dict'])
        Q2_targ.load_state_dict(checkpoint['Q2_targ_state_dict'])
        self.preprocessor_optimizer.load_state_dict(checkpoint['preprocessor_optimizer_state_dict'])
        self.Normal_optimizer.load_state_dict(checkpoint['Normal_optimizer_state_dict'])
        self.Q1_optimizer.load_state_dict(checkpoint['Q1_optimizer_state_dict'])
        self.Q2_optimizer.load_state_dict(checkpoint['Q2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self.log_alpha.requires_grad = True

        # Move optimizers to the correct device
        for optimizer in [self.preprocessor_optimizer, self.Normal_optimizer, self.Q1_optimizer, self.Q2_optimizer, self.alpha_optimizer]:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
