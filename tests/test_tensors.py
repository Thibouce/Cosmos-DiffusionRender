import numpy as np
import pytest
import torch

from cosmos_dr import tensors as T


def test_round_up_chunk_length():
    assert [T.round_up_chunk_length(n) for n in (1, 2, 9, 10, 57, 58)] == [1, 9, 9, 17, 57, 65]


def test_effective_chunk_length():
    assert T.effective_chunk_length(1, 57) == 1
    assert T.effective_chunk_length(10, 57) == 17
    assert T.effective_chunk_length(200, 57) == 57
    with pytest.raises(ValueError):
        T.effective_chunk_length(10, 56)


def test_chunk_ranges_cover_everything():
    assert T.chunk_ranges(120, 57) == [(0, 57), (57, 114), (114, 120)]
    assert T.chunk_ranges(1, 1) == [(0, 1)]


def test_validate_resolution():
    T.validate_resolution(704, 1280)
    with pytest.raises(ValueError):
        T.validate_resolution(700, 1280)


def test_pad_frames_repeats_last():
    x = torch.arange(3).float().view(1, 1, 3, 1, 1)
    y = T.pad_frames(x, 5)
    assert y.flatten().tolist() == [0, 1, 2, 2, 2]
    assert T.pad_frames(y, 2).flatten().tolist() == [0, 1]


def test_comfy_to_bcthw_range_and_shape():
    img = torch.rand(3, 32, 48, 3)
    out = T.comfy_to_bcthw(img, 64, 64, "stretch")
    assert out.shape == (1, 3, 3, 64, 64)
    assert out.min() >= -1 and out.max() <= 1


def test_comfy_to_bcthw_identity_values():
    img = torch.zeros(1, 16, 16, 3)
    img[..., 0] = 1.0
    out = T.comfy_to_bcthw(img, 16, 16)
    assert torch.allclose(out[0, 0], torch.ones(1, 16, 16))
    assert torch.allclose(out[0, 1], -torch.ones(1, 16, 16))


def test_crop_mode_keeps_aspect():
    # Left half black, right half white: a center crop to a square keeps both halves at the borders.
    img = torch.zeros(1, 16, 32, 3)
    img[:, :, 16:] = 1.0
    out = T.comfy_to_bcthw(img, 16, 16, "crop")
    assert out.shape == (1, 3, 1, 16, 16)
    assert out[0, 0, 0, :, 0].mean() < -0.9 and out[0, 0, 0, :, -1].mean() > 0.9


def test_single_channel_and_alpha_inputs():
    assert T.comfy_to_bcthw(torch.rand(1, 8, 8, 1), 16, 16).shape[1] == 3
    assert T.comfy_to_bcthw(torch.rand(1, 8, 8, 4), 16, 16).shape[1] == 3


def test_frames_roundtrip():
    frames = np.random.randint(0, 256, (2, 4, 4, 3), dtype=np.uint8)
    out = T.frames_to_comfy(frames)
    assert out.shape == (2, 4, 4, 3) and out.dtype == torch.float32
    assert torch.equal((out * 255).round().to(torch.uint8), torch.from_numpy(frames))


def test_build_data_batch_matches_upstream_collation():
    v = torch.zeros(1, 3, 9, 32, 48)
    b = T.build_data_batch({"rgb": v}, 32, 48, context_index=T.GBUFFER_INDEX_MAPPING["normal"])
    assert b["is_preprocessed"] is True
    assert b["context_index"].shape == (1, 1) and b["context_index"].item() == 3
    assert b["num_frames"].tolist() == [9.0]
    assert b["image_size"].tolist() == [[32, 48]]
    assert b["padding_mask"].shape == (1, 1, 32, 48)
    assert b["t5_text_embeddings"].shape == (1, 512, 1024)
    assert b["t5_text_mask"][0, 0] == 1 and b["t5_text_mask"][0, 1:].sum() == 0
