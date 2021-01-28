import pdb
from haiku import PRNGSequence
from typing import Any
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import optim
from models import apply_fast_slow_agent_model
from models import build_fast_slow_agent_model
from utils import categorical_importance_sampling_ratios
from utils import cross_entropy_loss_fn
from utils import vtrace

Tokenizer = Any


@jax.partial(jax.jit, static_argnums=(9, 10, 11, 12, 13, 14))
def gradient_step(
    agent_optimizer: optim.Optimizer,
    language_state: jnp.ndarray,
    vision_state: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    h_prev: jnp.ndarray,
    decoder_h_prev: jnp.ndarray,
    behavior_logits: jnp.ndarray,
    num_embeddings: int,
    embedding_dim: int,
    policy_eps: float = 0.1,
    baseline_eps: float = 0.5,
    entropy_eps: float = 1e-5,
    reconstruction_eps: float = 1.0,
) -> optim.Optimizer:
    def loss_fn(agent_params):
        (
            logits,
            v,
            reconstucted_lang,
            reconstructed_vision,
            *_,
        ) = apply_fast_slow_agent_model(
            agent_params,
            num_embeddings,
            embedding_dim,
            language_state,
            vision_state,
            h_prev,
            decoder_h_prev,
            False,
        )

        rhos = categorical_importance_sampling_ratios(
            logits[:-1], behavior_logits[:-1], actions[:-1]
        )

        errors, pg_advantages, q_estimate = vtrace()
        critic_loss = jnp.square(errors)

        log_pi_a = jnp.take_along_axis(
            jax.nn.log_softmax(logits[:-1]), actions[:-1], axis=-1
        )

        pg_advantage = jax.lax.stop_gradient(pg_advantage)
        pg_loss = jnp.mean(-log_pi_a * pg_advantage)

        entropy_loss = cross_entropy_loss_fn(logits, logits)

        language_reconstruction_loss = cross_entropy_loss_fn(
            language_state, reconstructed_lang
        )
        vision_reconstruction_loss = cross_entropy_loss_fn(
            vision_state, reconstructed_lang
        )

        regularized_loss = jnp.mean(
            policy_eps * pg_loss
            + baseline_eps * critic_loss
            + entropy_eps * entropy_loss
            + reconstruction_eps
            * (language_reconstruction_loss + vision_reconstruction_loss)
        )
        return loss

    grad = jax.grad(loss_fn)(agent_optimizer.target)
    return optimizer.apply_gradient(grad)


class Agent:
    def __init__(
        self,
        language_dim: int,
        vision_dim: Tuple[int, int],
        num_embeddings: int,
        embedding_dim: int,
        tokenizer: Tokenizer,
        discount: float = 0.9,
        hidden_size: int = 512,
        lr: float = 1e-5,
        policy_eps: float = 0.1,
        baseline_eps: float = 0.5,
        entropy_eps: float = 1e-5,
        reconstruction_eps: float = 1.0,
        seed: int = 0,
    ):
        self.rng = PRNGSequence(seed)

        input_dims = [
            (1, language_dim),
            (1, *vision_dim),
        ]
        agent_params = build_fast_slow_agent_model(
            input_dims, num_embeddings, embedding_dim, next(self.rng)
        )
        agent_optimizer = optim.Adam(learning_rate=lr).create(agent_params)
        self.agent_optimizer = jax.device_put(agent_optimizer)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.discount = discount
        self.policy_eps = policy_eps
        self.baseline_eps = baseline_eps
        self.entropy_eps = entropy_eps
        self.reconstruction_eps = reconstruction_eps

        self.tokenizer = tokenizer

        self.total_it = 0

    def get_tokens(self, timestep):
        encoded = self.tokenizer.encode(timestep.observation["TEXT"])
        return np.array(encoded.ids)

    def logits_and_hiddens(
        self,
        language: jnp.ndarray,
        vision: jnp.ndarray,
        h_prev: jnp.ndarray,
        decoder_h_prev: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        logits, *_, h, decoder_h = apply_fast_slow_agent_model(
            self.agent_optimizer.target,
            self.num_embeddings,
            self.embedding_dim,
            language,
            vision,
            h_prev,
            decoder_h_prev,
            training,
        )
        return logits, h, decoder_h

    def sample_action(
        self,
        rng: PRNGSequence,
        language: jnp.ndarray,
        vision: jnp.ndarray,
        h_prev: jnp.ndarray,
        decoder_h_prev: jnp.ndarray,
        trainin: bool = False,
    ) -> jnp.ndarray:
        logits, *_ = apply_fast_slow_agent_model(
            self.agent_optimizer.target,
            self.num_embeddings,
            self.embedding_dim,
            language,
            vision,
            h_prev,
            decoder_h_prev,
            training,
        )
        pi = jax.nn.softmax(logits, axis=-1)
        return random.sample(pi)

    def select_action(
        self,
        language: jnp.ndarray,
        vision: jnp.ndarray,
        h_prev: jnp.ndarray,
        decoder_h_prev: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        logits, *_ = apply_fast_slow_agent_model(
            self.agent_optimizer.target,
            self.num_embeddings,
            self.embedding_dim,
            language,
            vision,
            h_prev,
            decoder_h_prev,
            training,
        )
        return jnp.argmax(jnp.squeeze(logits))

    def train(self, replay_buffer, batch_size: int = 1):
        self.total_it += 1

        (
            language_state,
            vision_state,
            actions,
            rewards,
            discounts,
            extras,
        ) = replay_buffer.sample(next(self.rng), batch_size)

        h_prev = extras["memory_hiddens"]
        decoder_h_prev = extras["reconstruction_hiddens"]
        behavior_logits = extras["logits"]

        self.agent_optimizer = gradient_step(
            self.agent_optimizer,
            language_state,
            vision_state,
            actions,
            rewards,
            discounts,
            h_prev,
            decoder_h_prev,
            behavior_logits,
            self.num_embeddings,
            self.embedding_dim,
            self.policy_eps,
            self.baseline_eps,
            self.entropy_eps,
            self.reconstruction_eps,
        )
