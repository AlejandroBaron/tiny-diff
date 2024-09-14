import pytest
from torch import Tensor, randint, randn

B = 4
C = 3
H, W = 6, 6
CONTEXT_DIM = 32
DIFF_TIMESTEPS = 10
VAE_BASE_CHANNELS = 2
UNET_BASE_CHANNELS = 8
ATTN_HEAD_DIM = 5
NUM_ATTN_HEADS = 2


@pytest.fixture
def attention_context() -> Tensor:
    """Context vector for conditioning info (e.g. clip)."""
    return randn(B, 5, CONTEXT_DIM)


@pytest.fixture
def t_emb() -> Tensor:
    """Time embedding vector for diffusion processes."""
    return randint(0, DIFF_TIMESTEPS, size=(B,))
