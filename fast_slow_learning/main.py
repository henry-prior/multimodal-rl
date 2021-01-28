import dm_fast_mapping
import os

import numpy as np
from agent import Agent
from buffers import MultiModalReplayBuffer
from flax import linen as nn
from tokenizers import BertWordPieceTokenizer
from utils import eval_policy
from utils import FastSlowEnvWrapper


dir_path = os.path.dirname(os.path.realpath(__file__))

settings = dm_fast_mapping.EnvironmentSettings(
    seed=0, level_name="fast_slow/fast_map_three_objs"
)
env = dm_fast_mapping.load_from_docker(settings)
wrapped_env = FastSlowEnvWrapper(env)

temp_timestep = wrapped_env.reset()
language_dim = 108
vision_dim = temp_timestep.observation["RGB_INTERLEAVED"].shape
num_embeddings = 50
embedding_dim = 32

memory_hidden_dim = 512


tokenizer = BertWordPieceTokenizer(
    dir_path + "/embedding_files/bert-base-uncased-vocab.txt", lowercase=True
)
tokenizer.enable_padding(pad_id=3, length=language_dim, pad_token="[PAD]")

policy = Agent(
    language_dim=language_dim,
    vision_dim=vision_dim,
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    tokenizer=tokenizer,
)

# evaluations = [eval_policy(policy, settings)]
evaluations = []

episode_max_time = 1202

batch_size = 1
action_dim = 1
max_episodes = 100
train_steps = 1
eval_freq = 5
start_episodes = 1  # 10

max_size = 1000
action_dim = 1
replay_buffer = MultiModalReplayBuffer(
    max_size,
    episode_max_time,
    language_dim,
    vision_dim,
    action_dim,
    dict(
        memory_hiddens=(memory_hidden_dim,),
        reconstruction_hiddens=(embedding_dim,),
        logits=(wrapped_env.action_dim,),
    ),
)

timestep = wrapped_env.reset()
episode_reward = 0
episode_timesteps = 0
episode_num = 0

for t in range(int(max_episodes)):
    language_states = np.zeros((episode_max_time, language_dim))
    vision_states = np.zeros((episode_max_time, *vision_dim))
    actions = np.zeros((episode_max_time, action_dim))
    rewards = np.zeros((episode_max_time, 1))
    discounts = np.zeros((episode_max_time, 1))
    memory_hiddens = np.zeros((episode_max_time, memory_hidden_dim))
    reconstruction_hiddens = np.zeros((episode_max_time, embedding_dim))
    episode_logits = np.zeros((episode_max_time, wrapped_env.action_dim))

    h_prev = nn.LSTMCell().initialize_carry(next(policy.rng), (1,), memory_hidden_dim)
    decoder_h_prev = nn.LSTMCell().initialize_carry(
        next(policy.rng), (1,), embedding_dim
    )

    while not timestep.last():
        language_state = policy.get_tokens(timestep)
        vision_state = timestep.observation["RGB_INTERLEAVED"]

        language_states[episode_timesteps] = language_state
        vision_states[episode_timesteps] = vision_state
        memory_hiddens[episode_timesteps] = h_prev[1].squeeze()
        reconstruction_hiddens[episode_timesteps] = decoder_h_prev[1].squeeze()

        if t < start_episodes:
            logits = np.random.rand(wrapped_env.action_dim)
        else:
            logits, h_prev, decoder_h_prev = policy.logits_and_hiddens(
                language_state[None, :],
                vision_state[None, ...],
                h_prev,
                decoder_h_prev,
            )
        action = np.argmax(logits)

        timestep = wrapped_env.step(action)

        actions[episode_timesteps] = action
        rewards[episode_timesteps] = timestep.reward
        discounts[: episode_timesteps + 1] *= policy.discount
        episode_logits[episode_timesteps] = logits

        episode_reward += timestep.reward
        episode_timesteps += 1

    print(
        f"Total T: {t+1} Episode Num: {episode_num+1} "
        f" Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
    )

    timestep = wrapped_env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1

    replay_buffer.add(
        language_states,
        vision_states,
        actions,
        rewards,
        discounts,
        dict(
            memory_hiddens=memory_hiddens,
            reconstruction_hiddens=reconstruction_hiddens,
            logits=episode_logits,
        ),
    )

    if t >= start_episodes:
        for _ in range(train_steps):
            policy.train(replay_buffer, batch_size)

    # Evaluate episode
    if (t + 1) % eval_freq == 0:
        evaluations.append(eval_policy(policy, settings))
