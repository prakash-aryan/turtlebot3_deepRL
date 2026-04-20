# REDQ — Randomized Ensembled Double Q-learning.
#
# Same spirit as SAC (one actor, a critic, replay buffer, entropy bonus) but
# with two ideas that make it much more sample-efficient:
#
#   1. Keep a whole ensemble of critics (N of them, default 10) instead of
#      just two. When we compute the target Q, we pick a small random subset
#      of these critics (default 2) and take the minimum — this keeps the
#      value estimate from getting over-optimistic.
#
#   2. For every single environment step the robot takes, do many gradient
#      updates (G = 20 by default). This is called a high "update-to-data"
#      ratio. With the ensemble to stabilise things, those extra updates
#      don't blow up, and the policy learns from each real transition much
#      more thoroughly.
#
# The actor is a squashed Gaussian (outputs a mean and log-std, samples an
# action, squashes through tanh). An "alpha" temperature is learned
# automatically to keep the policy's entropy near a target value, which
# controls how much it explores.
#
# GPU note: the critic ensemble is implemented as a single VectorizedCritic
# with weight tensors of shape (N, in, out). All N critics run in one
# batched GPU call (torch.bmm) and are updated with one optimizer / one
# backward pass. Much faster on CUDA than N separate Linear modules.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.settings import (
    REDQ_ENSEMBLE_SIZE,
    REDQ_TARGET_SUBSET,
    REDQ_UTD_RATIO,
    REDQ_INIT_TEMPERATURE,
    REDQ_AUTOTUNE_ALPHA,
    REDQ_TARGET_ENTROPY,
    REDQ_BATCH_SIZE,
)
from .off_policy_agent import OffPolicyAgent, Network


# Clamps on the log-standard-deviation the actor is allowed to output —
# keeps the policy from collapsing to zero spread or exploding.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(Network):
    # Policy network: given the 44-dim state, produces a 2-dim action
    # (linear & angular velocity) by sampling a Gaussian and squashing it
    # through tanh so the result lives in [-1, 1].
    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        # Two heads: one predicts the mean of the Gaussian, the other its
        # log-std. Sampling from N(mean, std) gives us the stochastic action.
        self.mu_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)
        self.apply(super().init_weights)

    def forward(self, states, deterministic=False, with_logprob=True, visualize=False):
        # Two hidden ReLU layers.
        x1 = torch.relu(self.fa1(states))
        x2 = torch.relu(self.fa2(x1))

        # Predict mean + (clamped) log-std for the action distribution.
        mu = self.mu_head(x2)
        log_std = self.log_std_head(x2).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        # At test time we just use the mean (most-likely action).
        # At training time we sample so the policy can explore.
        if deterministic:
            u = mu
        else:
            noise = torch.randn_like(mu)
            u = mu + std * noise

        # Squash through tanh so the output fits the [-1, 1] action range.
        action = torch.tanh(u)

        # Compute the log-probability of the action under the current
        # policy. SAC / REDQ need this to (a) regularise the policy with
        # entropy, and (b) tune the temperature automatically. The log-prob
        # has to include a correction for the tanh squash.
        log_prob = None
        if with_logprob:
            log_prob_u = -0.5 * (((u - mu) / (std + 1e-8)) ** 2
                                 + 2 * log_std
                                 + math.log(2 * math.pi))
            log_prob_u = log_prob_u.sum(dim=-1, keepdim=True)
            log_prob = log_prob_u - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # Hook for the live network-visualiser (only used when ENABLE_VISUAL).
        if visualize and self.visual:
            self.visual.update_layers(states, action, [x1, x2], [self.fa1.bias, self.fa2.bias])
        return action, log_prob


class VectorizedCritic(Network):
    # All N critics packed into one module. Each parameter tensor has an
    # extra leading ensemble dim (shape N, in, out), so a single
    # torch.bmm(input, W) computes all N critics in one shot on the GPU.
    # On CUDA this is ~N× faster than running N separate Linear modules
    # sequentially, which is critical for REDQ because we do G × N
    # critic forward/backward passes every env step.
    def __init__(self, name, state_size, action_size, hidden_size, N):
        super().__init__(name)
        self.N = N
        in_dim = state_size + action_size
        # Three linear layers, each stacked across the ensemble dim.
        self.W1 = nn.Parameter(torch.empty(N, in_dim,      hidden_size))
        self.b1 = nn.Parameter(torch.zeros(N, 1,           hidden_size))
        self.W2 = nn.Parameter(torch.empty(N, hidden_size, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(N, 1,           hidden_size))
        self.W3 = nn.Parameter(torch.empty(N, hidden_size, 1))
        self.b3 = nn.Parameter(torch.zeros(N, 1,           1))
        # Xavier init per ensemble slice.
        for W in (self.W1, self.W2, self.W3):
            for i in range(N):
                nn.init.xavier_uniform_(W[i])

    def forward(self, states, actions):
        # states: [B, S], actions: [B, A]  →  returns [N, B, 1]
        x = torch.cat([states, actions], dim=-1)            # [B, S+A]
        x = x.unsqueeze(0).expand(self.N, -1, -1)           # [N, B, S+A]
        x = torch.relu(torch.bmm(x, self.W1) + self.b1)     # [N, B, H]
        x = torch.relu(torch.bmm(x, self.W2) + self.b2)     # [N, B, H]
        q = torch.bmm(x, self.W3) + self.b3                 # [N, B, 1]
        return q


class REDQ(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        # Hyperparameters specific to REDQ. N is how many critics we keep,
        # M is how many of them we use to build the target (pessimistic
        # minimum over a random subset), G is how many gradient updates
        # we do per env step.
        self.ensemble_size = REDQ_ENSEMBLE_SIZE         # N
        self.target_subset = REDQ_TARGET_SUBSET         # M (<= N)
        self.utd_ratio     = REDQ_UTD_RATIO             # G

        # A bigger batch keeps the GPU busier. The base class stores the
        # default BATCH_SIZE; we override with the REDQ-specific one.
        self.batch_size = REDQ_BATCH_SIZE

        # No ε-greedy — the Gaussian policy does its own exploration.
        self.epsilon = None
        self.epsilon_decay = 1.0

        # Cache last losses / alpha so they're easy to log each episode.
        self.last_actor_loss = 0.0
        self.last_alpha = REDQ_INIT_TEMPERATURE

        # One actor, one optimizer.
        self.actor = self.create_network(SquashedGaussianActor, 'actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        # Critic ensemble: one VectorizedCritic holding all N critics as a
        # single GPU tensor. Same for the target copy. One optimizer covers
        # all of them — a single backward() updates all N at once.
        self.critic        = self._build_vectorized_critic('critic')
        self.critic_target = self._build_vectorized_critic('target_critic')
        self.hard_update(self.critic_target, self.critic)
        self.critic_optimizer = self.create_optimizer(self.critic)

        # Temperature α weights the entropy bonus in both the actor loss
        # and the TD target. With auto-tuning, we keep a learnable
        # log_alpha and nudge it so the policy's average entropy stays
        # near `target_entropy` (a heuristic default of -|action_size|).
        self.autotune_alpha = REDQ_AUTOTUNE_ALPHA
        if self.autotune_alpha:
            self.log_alpha = torch.tensor(
                math.log(REDQ_INIT_TEMPERATURE),
                device=self.device, requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], self.learning_rate)
            self.target_entropy = (REDQ_TARGET_ENTROPY
                                   if REDQ_TARGET_ENTROPY is not None
                                   else -float(self.action_size))
        else:
            self.log_alpha = torch.tensor(math.log(REDQ_INIT_TEMPERATURE), device=self.device)

    def _build_vectorized_critic(self, name):
        # Helper so we construct live and target critics identically.
        net = VectorizedCritic(
            name, self.state_size, self.action_size,
            self.hidden_size, self.ensemble_size,
        ).to(self.device)
        self.networks.append(net)
        return net

    # Convenience: α is always exp(log_alpha) because α must stay > 0.
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, state, is_training, step, visualize=False):
        # Called every env step. During training we sample from the policy
        # (exploration); during testing we take the deterministic mean
        # (exploitation).
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state, deterministic=not is_training,
                                   with_logprob=False, visualize=visualize)
        return action.squeeze(0).cpu().numpy().tolist()

    def get_action_random(self):
        # Used during the first OBSERVE_STEPS steps (pure random exploration
        # to seed the replay buffer before any training happens).
        return [float(np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0))] * self.action_size

    def _train(self, replaybuffer):
        # This is the "update-to-data = G" trick: on every env step the
        # base class calls this once, and we do G gradient updates here
        # (each on a fresh mini-batch) instead of the usual one. This is
        # what makes REDQ learn so much more out of each real transition.
        last_critic_loss = 0.0
        last_actor_loss  = self.last_actor_loss
        for _ in range(self.utd_ratio):
            batch = replaybuffer.sample(self.batch_size)
            s, a, r, ns, d = batch
            # Upload the batch to GPU in one shot — everything that follows
            # stays on-device until the losses come back to CPU for logging.
            s  = torch.from_numpy(s).to(self.device,  non_blocking=True)
            a  = torch.from_numpy(a).to(self.device,  non_blocking=True)
            r  = torch.from_numpy(r).to(self.device,  non_blocking=True)
            ns = torch.from_numpy(ns).to(self.device, non_blocking=True)
            d  = torch.from_numpy(d).to(self.device,  non_blocking=True)
            last_critic_loss, last_actor_loss = self.train(s, a, r, ns, d)
            self.iteration += 1
        return [last_critic_loss, last_actor_loss]

    def train(self, state, action, reward, state_next, done):
        # --- Step 1: build the TD target ---------------------------------
        # Ask the actor what it would do in the next state, then ask the
        # target critics how good that action looks. We only trust a small
        # RANDOM SUBSET of size M, and take the minimum — this is the
        # pessimistic "double Q" idea, extended to an ensemble, which is
        # what keeps the Q-values from blowing up under high UTD.
        with torch.no_grad():
            action_next, logprob_next = self.actor(state_next, with_logprob=True)
            # target critic forward once → [N, B, 1]
            q_next_all = self.critic_target(state_next, action_next)
            # sample M out of N critics (on-device index tensor, no python loop)
            subset = torch.randperm(self.ensemble_size, device=self.device)[: self.target_subset]
            q_next = q_next_all[subset].min(dim=0).values            # [B, 1]
            # Standard SAC target: future value minus the entropy term so
            # the policy is rewarded for staying flexible.
            target = reward + (1.0 - done) * self.discount_factor * (
                q_next - self.alpha * logprob_next
            )

        # --- Step 2: update every critic in the ensemble -----------------
        # Single forward + single backward — the ensemble is packed into
        # one module, so all N critics' gradients are computed in one pass.
        q_pred = self.critic(state, action)                      # [N, B, 1]
        target_bcast = target.unsqueeze(0).expand_as(q_pred)     # [N, B, 1]
        critic_loss = F.mse_loss(q_pred, target_bcast)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        # --- Step 3: update the actor -----------------------------------
        # Sample a fresh action from the current policy, score it with the
        # MEAN Q across ALL critics, and push the policy toward actions the
        # critics like — minus the entropy bonus so it doesn't collapse to
        # a single-point policy.
        action_pi, logprob_pi = self.actor(state, with_logprob=True)
        q_pi_all = self.critic(state, action_pi).mean(dim=0)     # [B, 1]
        actor_loss = (self.alpha.detach() * logprob_pi - q_pi_all).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()
        self.last_actor_loss = actor_loss.detach().cpu()

        # --- Step 4: auto-tune the temperature α ------------------------
        # If the policy's entropy drifts above the target, raise α so the
        # entropy term matters more; below target, lower α. This replaces
        # the fragile manual tuning SAC originally needed.
        if self.autotune_alpha:
            alpha_loss = -(self.log_alpha
                           * (logprob_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.last_alpha = self.alpha.detach().cpu().item()

        # --- Step 5: slowly update the target ensemble ------------------
        # Polyak average. Because the live and target critics are both
        # single VectorizedCritic modules, soft_update walks all N sets
        # of weights in one go.
        self.soft_update(self.critic_target, self.critic, self.tau)

        return [critic_loss.detach().cpu(), self.last_actor_loss]

    def get_model_parameters(self):
        # Appends REDQ-specific knobs (N, M, G, current α) to the base
        # parameter string so the logger records them.
        base = super().get_model_parameters()
        return (base
                + f", N={self.ensemble_size}, M={self.target_subset},"
                + f" G={self.utd_ratio}, batch={self.batch_size},"
                + f" α={self.last_alpha:.3f}")
