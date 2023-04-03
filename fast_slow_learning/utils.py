import dm_fast_mapping

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
from typing import Tuple


ACTION_TYPES = [
    "HAND_GRIP",
    "HAND_ROTATE_AROUND_FORWARD",
    "USE_HELD_OBJECT",
    "STRAFE_LEFT_RIGHT",
    "HAND_ROTATE_AROUND_UP",
    "LANGUAGE",
    "LOOK_LEFT_RIGHT",
    "HAND_PUSH_PULL",
    "MOVE_BACK_FORWARD",
    "HAND_ROTATE_AROUND_RIGHT",
    "LOOK_DOWN_UP",
]
ACTIONS_DICT = dict(zip(ACTION_TYPES, len(ACTION_TYPES) * [0.0]))
ACTIONS = [
    ACTIONS_DICT,  # NOOP
    dict(ACTIONS_DICT, MOVE_BACK_FORWARD=1.0),  # MOVE_FORWARD(1)
    dict(ACTIONS_DICT, MOVE_BACK_FORWARD=-1.0),  # MOVE_FORWARD(-1)
    dict(ACTIONS_DICT, STRAFE_LEFT_RIGHT=1.0),  # MOVE_RIGHT(1)
    dict(ACTIONS_DICT, STRAFE_LEFT_RIGHT=-1.0),  # MOVE_RIGHT(-1)
    dict(ACTIONS_DICT, LOOK_LEFT_RIGHT=1.0),  # LOOK_RIGHT(1)
    dict(ACTIONS_DICT, LOOK_LEFT_RIGHT=-1.0),  # LOOK_RIGHT(-1)
    dict(ACTIONS_DICT, LOOK_DOWN_UP=-1.0),  # LOOK_DOWN(1)
    dict(ACTIONS_DICT, LOOK_DOWN_UP=1.0),  # LOOK_DOWN(-1)
    dict(ACTIONS_DICT, STRAFE_LEFT_RIGHT=0.05),  # MOVE_RIGHT(0.05)
    dict(ACTIONS_DICT, STRAFE_LEFT_RIGHT=-0.05),  # MOVE_RIGHT(-0.05)
    dict(ACTIONS_DICT, LOOK_DOWN_UP=-0.03),  # LOOK_DOWN(0.03)
    dict(ACTIONS_DICT, LOOK_DOWN_UP=0.03),  # LOOK_DOWN(-0.03)
    dict(ACTIONS_DICT, LOOK_LEFT_RIGHT=0.2),  # LOOK_RIGHT(0.2)
    dict(ACTIONS_DICT, LOOK_LEFT_RIGHT=-0.2),  # LOOK_RIGHT(-0.2)
    dict(ACTIONS_DICT, LOOK_LEFT_RIGHT=0.05),  # LOOK_RIGHT(0.05)
    dict(ACTIONS_DICT, LOOK_LEFT_RIGHT=-0.05),  # LOOK_RIGHT(-0.05)
    dict(ACTIONS_DICT, HAND_GRIP=1.0),  # GRAB
    dict(ACTIONS_DICT, HAND_GRIP=1.0, MOVE_BACK_FORWARD=1.0),  # GRAB + MOVE_FORWARD(1)
    dict(
        ACTIONS_DICT, HAND_GRIP=1.0, MOVE_BACK_FORWARD=-1.0
    ),  # GRAB + MOVE_FORWARD(-1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, STRAFE_LEFT_RIGHT=1.0),  # GRAB + MOVE_RIGHT(1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, STRAFE_LEFT_RIGHT=-1.0),  # GRAB + MOVE_RIGHT(-1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, LOOK_LEFT_RIGHT=1.0),  # GRAB + LOOK_RIGHT(1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, LOOK_LEFT_RIGHT=-1.0),  # GRAB + LOOK_RIGHT(-1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, LOOK_DOWN_UP=-1.0),  # GRAB + LOOK_DOWN(1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, LOOK_DOWN_UP=1.0),  # GRAB + LOOK_DOWN(-1)
    dict(
        ACTIONS_DICT, HAND_GRIP=1.0, STRAFE_LEFT_RIGHT=0.05
    ),  # GRAB + MOVE_RIGHT(0.05)
    dict(
        ACTIONS_DICT, HAND_GRIP=1.0, STRAFE_LEFT_RIGHT=-0.05
    ),  # GRAB + MOVE_RIGHT(-0.05)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, LOOK_DOWN_UP=-0.03),  # GRAB + LOOK_DOWN(0.03)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, LOOK_DOWN_UP=0.03),  # GRAB + LOOK_DOWN(-0.03)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, LOOK_LEFT_RIGHT=0.2),  # GRAB + LOOK_RIGHT(0.2)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, LOOK_LEFT_RIGHT=-0.2),  # GRAB + LOOK_RIGHT(-0.2)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, LOOK_LEFT_RIGHT=0.05),  # GRAB + LOOK_RIGHT(0.05)
    dict(
        ACTIONS_DICT, HAND_GRIP=1.0, LOOK_LEFT_RIGHT=-0.05
    ),  # GRAB + LOOK_RIGHT(-0.05)
    dict(
        ACTIONS_DICT, HAND_GRIP=1.0, HAND_ROTATE_AROUND_RIGHT=1.0
    ),  # GRAB + SPIN_RIGHT(1)
    dict(
        ACTIONS_DICT, HAND_GRIP=1.0, HAND_ROTATE_AROUND_RIGHT=-1.0
    ),  # GRAB + SPIN_RIGHT(1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, HAND_ROTATE_AROUND_UP=1.0),  # GRAB + SPIN_UP(1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, HAND_ROTATE_AROUND_UP=-1.0),  # GRAB + SPIN_UP(1)
    dict(
        ACTIONS_DICT, HAND_GRIP=1.0, HAND_ROTATE_AROUND_FORWARD=1.0
    ),  # GRAB + SPIN_FORWARD(1)
    dict(
        ACTIONS_DICT, HAND_GRIP=1.0, HAND_ROTATE_AROUND_FORWARD=-1.0
    ),  # GRAB + SPIN_FORWARD(1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, HAND_PUSH_PULL=1.0),  # GRAB + PULL(1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, HAND_PUSH_PULL=-1.0),  # GRAB + PULL(1)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, HAND_PUSH_PULL=0.5),  # GRAB + PULL(0.5)
    dict(ACTIONS_DICT, HAND_GRIP=1.0, HAND_PUSH_PULL=-0.5),  # GRAB + PULL(-0.5)
    dict(ACTIONS_DICT, HAND_PUSH_PULL=0.5),  # PULL(0.5)
    dict(ACTIONS_DICT, HAND_PUSH_PULL=-0.5),  # PULL(-0.5)
]


class FastSlowEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.action_dim = 46

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def action_spec(self):
        return self.env.action_spec()

    def step(self, action: int):
        assert 0 <= action < 46, (
            "The action space of this environment is a discretization and takes "
            "values in the range 0-45."
        )
        action_dict = ACTIONS[action]
        return self.env.step(action_dict)


def eval_policy(policy, settings, eval_episodes=10):
    env = dm_fast_mapping.load_from_docker(settings)
    env = FastSlowEnvWrapper(env)

    avg_reward = 0.0
    count = 0
    for _ in range(eval_episodes):
        timestep = env.reset()
        h_prev = nn.LSTMCell.initialize_carry(next(policy.rng), (1,), 512)
        decoder_h_prev = nn.LSTMCell.initialize_carry(next(policy.rng), (1,), 32)
        while not timestep.last():
            vision_state = timestep.observation["RGB_INTERLEAVED"]
            language_state = policy.get_tokens(timestep)

            logits, h_prev, decoder_h_prev = policy.logits_and_hiddens(
                language_state[None, :],
                vision_state[None, ...],
                h_prev,
                decoder_h_prev,
                training=False,
            )
            action = jnp.argmax(logits)
            timestep = env.step(action)
            avg_reward += timestep.reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def cross_entropy_loss_fn(logits_p: jnp.ndarray, logits_q: jnp.ndarray) -> jnp.ndarray:
    probs = jax.nn.softmax(logits_p)
    logprobs = jax.nn.log_softmax(logits_q)
    return -jnp.sum(probs * logprobs, axis=-1)


def categorical_importance_sampling_ratios(
    pi_logits: jnp.ndarray, mu_logits: jnp.ndarray, actions: jnp.ndarray
) -> jnp.ndarray:
    """
    Importance sampling ratios. Calculated in log space for numerical stability.


    Args:
        pi_logits: logits of the target policy
        mu_logits: logits of the behavior policy
        actions: the actions taken by the behavior policy at each index
    """

    log_pi_a = jnp.take_along_axis(
        jax.nn.log_softmax(pi_logits), actions[..., None], axis=-1
    )
    log_mu_a = jnp.take_along_axis(
        jax.nn.log_softmax(mu_logits), actions[..., None], axis=-1
    )
    rho = jnp.exp(log_pi_a - log_mu_a)
    return rho


def vtrace(
    values_prev: jnp.ndarray,
    values: jnp.ndarray,
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    rhos: jnp.ndarray,
    lambda_: float = 1.0,
    rho_clip: float = 1.0,
) -> jnp.ndarray:
    c_t = jnp.minimum(1.0, rhos) * lambda_
    clipped_rhos = jnp.minimum(rho_clip, rhos)

    weighted_td_errors = clipped_rhos * (rewards + discounts * values - value_prev)

    current_error = 0.0
    errors = []
    for i in jnp.arange(values.shape[0] - 1, -1, -1):
        current_error = weighted_td_errors[i] + discounts[i] * c_t[i] * current_error
        errors.insert(0, current_error)

    target_prev = jnp.array(errors) + values_prev
    target_prev = jax.lax.stop_gradient(target_prev)

    q_bootstrap = jnp.concatenate(
        [
            lambda_ * target_prev[1:] + (1 - lambda_) * values_prev[1:],
            values[-1:],
        ],
        axis=0,
    )
    q_estimate = rewards + discounts * q_bootstrap
    pg_advantages = clipped_rhos * (q_estimate - values_prev)
    return errors, pg_advantages, q_estimate


def rescale_dims(shape: Sequence[int], along: Tuple, scale: float) -> Sequence[int]:
    out_shape = [s * scale if idx in along else s for idx, s in enumerate(shape)]
    return out_shape
