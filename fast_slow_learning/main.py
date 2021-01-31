import dm_fast_mapping
import os

import numpy as np
from agent import Agent
from buffers import MultiModalReplayBuffer
from flax import linen as nn
from tokenizers import BertWordPieceTokenizer
from utils import eval_policy
from utils import FastSlowEnvWrapper
import argparse


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level_name", default="fast_slow/fast_map_three_objs")
    parser.add_argument("--episode_max_time", type=int, default=1202)
    parser.add_argument("--language_dim", type=int, default=108)
    parser.add_argument("--num_embeddings", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--memory_hidden_dim", type=int, default=512)
    parser.add_argument("--action_dim", type=int, default=1)
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--train_steps", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--start_episodes", type=int, default=10)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    settings = dm_fast_mapping.EnvironmentSettings(seed=0, level_name=args.level_name)
    env = dm_fast_mapping.load_from_docker(settings)
    wrapped_env = FastSlowEnvWrapper(env)

    temp_timestep = wrapped_env.reset()
    vision_dim = temp_timestep.observation["RGB_INTERLEAVED"].shape

    tokenizer = BertWordPieceTokenizer(
        DIR_PATH + "/embedding_files/bert-base-uncased-vocab.txt", lowercase=True
    )
    tokenizer.enable_padding(pad_id=3, length=args.language_dim, pad_token="[PAD]")

    policy = Agent(
        language_dim=args.language_dim,
        vision_dim=vision_dim,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        memory_hidden_dim=args.memory_hidden_dim,
        tokenizer=tokenizer,
    )

    evaluations = [eval_policy(policy, settings)]

    replay_buffer = MultiModalReplayBuffer(
        args.buffer_size,
        args.episode_max_time,
        args.language_dim,
        vision_dim,
        args.action_dim,
        dict(
            memory_hiddens=(args.memory_hidden_dim,),
            reconstruction_hiddens=(args.embedding_dim,),
            logits=(wrapped_env.action_dim,),
        ),
    )

    timestep = wrapped_env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_episodes)):
        language_states = np.zeros((episode_max_time, args.language_dim))
        vision_states = np.zeros((episode_max_time, *vision_dim))
        actions = np.zeros((episode_max_time, args.action_dim))
        rewards = np.zeros((episode_max_time, 1))
        discounts = np.zeros((episode_max_time, 1))
        memory_hiddens = np.zeros((episode_max_time, args.memory_hidden_dim))
        reconstruction_hiddens = np.zeros((episode_max_time, args.embedding_dim))
        episode_logits = np.zeros((episode_max_time, wrapped_env.action_dim))

        h_prev = nn.LSTMCell().initialize_carry(
            next(policy.rng), (1,), args.memory_hidden_dim
        )
        decoder_h_prev = nn.LSTMCell().initialize_carry(
            next(policy.rng), (1,), args.embedding_dim
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
                policy.train(replay_buffer, args.batch_size)

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_policy(policy, settings))
