"""Test image packing from GDK Image -> CHW uint8."""

from __future__ import annotations

import numpy as np
import pytest

from examples.agibot_real import constants
from examples.agibot_real.env import AgibotRealEnvironment
from examples.agibot_real.tests.fakes.agibot_gdk import FakeImage


def _solid_rgb_image(h: int, w: int, color: tuple[int, int, int]) -> FakeImage:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    return FakeImage(data=arr.tobytes(), width=w, height=h, encoding="rgb8")


def test_image_to_chw_shape_and_dtype():
    img = _solid_rgb_image(480, 640, (10, 20, 30))
    out = AgibotRealEnvironment._image_to_chw(img, *constants.TARGET_HW)
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.uint8


def test_image_to_chw_preserves_color_for_uniform_image():
    # A solid-colored image with no padding should round-trip cleanly.
    img = _solid_rgb_image(224, 224, (10, 20, 30))
    out = AgibotRealEnvironment._image_to_chw(img, 224, 224)
    # CHW: out[0] is R, out[1] is G, out[2] is B
    assert out[0].mean() == 10
    assert out[1].mean() == 20
    assert out[2].mean() == 30


def test_image_to_chw_swaps_bgr_to_rgb():
    img = _solid_rgb_image(224, 224, (10, 20, 30))
    img.encoding = "bgr8"
    out = AgibotRealEnvironment._image_to_chw(img, 224, 224)
    # After BGR->RGB swap, channel 0 (R) should hold what was originally
    # encoded as B (third byte = 30).
    assert out[0].mean() == 30
    assert out[2].mean() == 10


def test_image_to_chw_rejects_unknown_encoding():
    img = _solid_rgb_image(224, 224, (1, 2, 3))
    img.encoding = "yuv422"
    with pytest.raises(ValueError, match="unsupported image encoding"):
        AgibotRealEnvironment._image_to_chw(img, 224, 224)
