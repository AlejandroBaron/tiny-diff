import pytest
from torch import Tensor, randn

from tests.conftest import ATTN_HEAD_DIM, CONTEXT_DIM, NUM_ATTN_HEADS, B, C, H, W
from tiny_diff.modules.attention import (
    CrossVisualAttention,
    VisualAttention,
)


@pytest.fixture
def q() -> Tensor:
    """Query tensor."""
    return randn(B, C, H, W)


@pytest.fixture
def k() -> Tensor:
    """Key tensor."""
    return randn(B, C, H * 2, W * 2)


@pytest.fixture
def v() -> Tensor:
    """Value tensor."""
    return randn(B, C, H * 2, W * 2)


@pytest.fixture
def va_layer() -> VisualAttention:
    """Visual attention layer."""
    return VisualAttention(channels=C, head_dim=ATTN_HEAD_DIM, num_heads=NUM_ATTN_HEADS)


@pytest.fixture
def cva_layer() -> CrossVisualAttention:
    """Conditional Visual Attention layer."""
    return CrossVisualAttention(
        channels=C,
        context_dim=CONTEXT_DIM,
        head_dim=ATTN_HEAD_DIM,
        num_heads=NUM_ATTN_HEADS,
    )


def test_visual_attention(q: Tensor, k: Tensor, v: Tensor, va_layer: VisualAttention):
    """Tests that the output shape matches."""
    result = va_layer(q, k, v)
    assert result.shape == q.shape


def test_cross_visual_attention(
    q: Tensor, attention_context, cva_layer: VisualAttention
):
    """Tests that the output shape matches."""
    result = cva_layer(q, attention_context)
    assert result.shape == q.shape
