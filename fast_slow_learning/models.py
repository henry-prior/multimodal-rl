from haiku import PRNGSequence
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import random


class LatentEncoder(nn.Module):
    @nn.compact
    def __call__(self, hidden, residual, vision, language):
        x = jnp.concatenate([hidden, residual, vision, language], axis=-1)
        x = nn.Dense(features=256)(x)
        return x


class ResNetBlock(nn.Module):
    channels: int
    conv: nn.Module
    activation: Callable = nn.relu
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        y = self.conv(self.channels, (3, 3), self.strides)(x)
        y = self.activation(y)
        y = self.conv(self.channels, (3, 3), self.strides)(y)

        return self.activation(residual + y)


class ResNet(nn.Module):
    block_sizes: Sequence[int]
    channels_per_block: Sequence[int]
    output_dim: Union[int, Tuple[int, int]]
    conv: nn.Module = nn.Conv
    activation: Callable = nn.relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = True):
        assert len(self.block_sizes) == len(
            self.channels_per_block
        ), "Arguments `block_sizes` and `channels_per_block` must be the same length."
        conv = jax.partial(self.conv, use_bias=False, dtype=self.dtype)

        for block_size, n_channels in zip(self.block_sizes, self.channels_per_block):
            x = conv(n_channels, (3, 3), (1, 1))(x)
            # x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
            for _ in range(block_size):
                x = ResNetBlock(n_channels, conv=conv, activation=self.activation)(x)

        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(self.output_dim, dtype=self.dtype)(x)
        x = self.activation(x)
        return x


class ResNetTransposeBlock(nn.Module):
    channels: int
    conv: nn.Module
    activation: Callable = nn.relu
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.channels, (3, 3), self.strides)(x)
        y = self.activation(y)
        y = self.conv(self.channels, (3, 3), self.strides)(y)

        return self.activation(residual + y)


class ResNetTranspose(nn.Module):
    block_sizes: Sequence[int]
    channels_per_block: Sequence[int]
    output_dim: Union[int, Sequence[int]]
    conv: nn.Module = nn.ConvTranspose
    activation: Callable = nn.relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        conv = jax.partial(self.conv, use_bias=False, dtype=self.dtype)

        for block_size, n_channels in zip(self.block_sizes, self.channels_per_block):
            x = conv(n_channels, (3, 3), (1, 1))(x)
            # x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
            for _ in range(block_size):
                x = ResNetTransposeBlock(
                    n_channels, conv=conv, activation=self.activation
                )(x)

        x = conv(3, (3, 3), (1, 1))(x)

        return x


class LanguageEncoder(nn.Module):
    num_embeddings: int
    embedding_dim: int
    num_heads: int = 1
    qkv_features: int = 16
    out_features: int = 32

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = x.astype(jnp.int32)
        x = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.embedding_dim,
            name="embed",
        )(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.out_features,
            use_bias=False,
            deterministic=not training,
        )(x)
        return x


class DCEM(nn.Module):
    head_dim: int = 256
    num_heads: int = 3
    out_features: int = 256

    """
    The Dual-coded Episodic Memory from figure 9 in `Grounded Language Learning
    Fast and Slow (Hill et al., 2021) <https://arxiv.org/pdf/2009.01719.pdf>`_.

    Uses a transformer-based approach to attend to the language and image embeddings
    """

    @nn.compact
    def __call__(self, language, vision, hidden):
        input_q = jnp.concatenate([language, hidden], axis=-1)

        query = nn.DenseGeneral(features=(self.num_heads, self.head_dim), name="query")(
            input_q
        )
        key = nn.DenseGeneral(features=(self.num_heads, self.head_dim), name="key")(
            language
        )
        value = nn.DenseGeneral(
            features=(self.num_heads, self.head_dim), name="memory_value"
        )(vision)

        x = nn.dot_product_attention(query, key, value)

        out = nn.DenseGeneral(features=self.out_features, axis=(-2, -1), name="out")(x)
        return out


class FastSlowAgent(nn.Module):
    num_embeddings: int
    embedding_dim: int
    block_sizes: Sequence[int] = (2, 2, 2)
    channels_per_block: Sequence[int] = (16, 32, 32)
    visual_embedding_dim: int = 256
    num_discrete_actions: int = 46

    """
    The whole agent architecture from figure 9 in `Grounded Language Learning
    Fast and Slow (Hill et al., 2021) <https://arxiv.org/pdf/2009.01719.pdf>`_.

    Contains a policy and value head as well as a reconstruction head for each
    of the state inputs (language and vision).
    """

    @nn.compact
    def __call__(self, language, vision, h_prev, decoder_h_prev, training: bool = True):
        embedded_language = LanguageEncoder(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )(language)
        embedded_vision = ResNet(
            block_sizes=self.block_sizes,
            channels_per_block=self.channels_per_block,
            output_dim=self.visual_embedding_dim,
        )(vision, training=training)

        residual = DCEM()(embedded_language, embedded_vision, h_prev[1])
        e = LatentEncoder()(h_prev[1], residual, embedded_vision, embedded_language)
        h, x = nn.LSTMCell()(h_prev, e)

        policy_latent = nn.Dense(features=256)(x)
        logits = nn.Dense(features=self.num_discrete_actions)(policy_latent)

        value_latent = nn.Dense(features=256)(x)
        v = nn.Dense(features=1)(value_latent)

        decoder_h, reconstructed_lang = nn.LSTMCell()(decoder_h_prev, e)
        e = nn.Dense(features=72 * 96 * 32)(e)
        e = nn.relu(e)
        e = jnp.reshape(e, (-1, 72, 96, 32))
        reconstructed_vision = ResNetTranspose(
            block_sizes=self.block_sizes,
            channels_per_block=self.channels_per_block[::-1],
            output_dim=(72, 96, 3),
        )(e)

        return logits, v, reconstructed_lang, reconstructed_vision, h, decoder_h


class ValueAndReconstructionHead(nn.Module):
    block_sizes: Sequence[int] = (2, 2, 2)
    channels_per_block: Sequence[int] = (16, 32, 32)

    @nn.compact
    def __call__(self, x, e, decoder_h_prev):
        value_latent = nn.Dense(features=256)(x)
        v = nn.Dense(features=1)(value_latent)

        decoder_h, reconstructed_lang = nn.LSTMCell()(decoder_h_prev, e)
        e = nn.Dense(features=72 * 96 * 32)(e)
        e = nn.relu(e)
        e = jnp.reshape(e, (-1, 72, 96, 32))
        reconstructed_image = ResNetTranspose(
            block_sizes=self.block_sizes,
            channels_per_block=self.channels_per_block,
            output_dim=(72, 96, 3),
        )(e)

        return v, reconstructed_lang, reconstructed_vision, decoder_h


def build_fast_slow_agent_model(
    input_shapes: Sequence[Tuple[int, int]],
    memory_hidden_dim: int,
    decoder_hidden_dim: int,
    num_embeddings: int,
    embedding_dim: int,
    init_rng: PRNGSequence,
) -> FrozenDict:
    init_rng, hidden_core_rng, hidden_decode_rng = jax.random.split(init_rng, 3)
    init_batch = [jnp.ones(shape, jnp.float32) for shape in input_shapes]
    fast_slow_agent = FastSlowAgent(
        num_embeddings=num_embeddings, embedding_dim=embedding_dim
    )
    init_variables = fast_slow_agent.init(
        init_rng,
        *init_batch,
        nn.LSTMCell.initialize_carry(hidden_core_rng, (1,), memory_hidden_dim),
        nn.LSTMCell.initialize_carry(hidden_decode_rng, (1,), decoder_hidden_dim),
    )
    return init_variables


@jax.partial(jax.jit, static_argnums=(1, 2, 7))
def apply_fast_slow_agent_model(
    params: FrozenDict,
    num_embeddings: int,
    embedding_dim: int,
    language: jnp.ndarray,
    vision: jnp.ndarray,
    h_prev: jnp.ndarray,
    decoder_h_prev: jnp.ndarray,
    training: bool,
) -> jnp.ndarray:
    return FastSlowAgent(
        num_embeddings=num_embeddings, embedding_dim=embedding_dim
    ).apply(params, language, vision, h_prev, decoder_h_prev, training)
