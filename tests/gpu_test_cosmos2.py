#!/usr/bin/env python3
"""
GPU Test: Cosmos 2.0 (nvidia/Cosmos-Predict2-2B-Video2World) integration.

Tests:
1. Tiny mode: Cosmos2PredictWrapper with random weights → full MimicVideo forward + sample
2. Pretrained mode: Load real Cosmos 2.0 weights (individually, no pipeline safety checker)
3. Training loop: 5 steps with tiny Cosmos 2.0 + MimicVideo

Usage: python tests/gpu_test_cosmos2.py
"""

import sys
import traceback
import torch
import torch.nn.functional as F

RESULTS = []


def log(msg, ok=True):
    status = "✓" if ok else "✗"
    print(f"  {status} {msg}")
    RESULTS.append((msg, ok))


def test_tiny_forward():
    """Test Cosmos2PredictWrapper tiny mode → MimicVideo forward."""
    print("\n=== Test 1: Cosmos2 Tiny Mode Forward ===")
    from mimic_video.cosmos_predict import Cosmos2PredictWrapper
    from mimic_video.mimic_video import MimicVideo

    device = torch.device("cuda")
    wrapper = Cosmos2PredictWrapper(random_weights=True, tiny=True, extract_layer=0)

    model = MimicVideo(
        dim=32,
        video_predict_wrapper=wrapper,
        action_chunk_len=4,
        dim_action=32,
        dim_joint_state=64,
        depth=1,
        dim_head=16,
        heads=2,
        num_video_viewpoints=1,
    ).to(device)

    # Forward pass
    B = 2
    video = torch.randn(B, 1, 1, 3, 32, 32, device=device)  # (B, V, T, C, H, W)
    actions = torch.randn(B, 4, 32, device=device)
    joint_state = torch.randn(B, 64, device=device)
    action_mask = torch.zeros(B, 32, dtype=torch.bool, device=device)
    action_mask[:, :7] = True

    loss = model(
        video=video,
        actions=actions,
        joint_state=joint_state,
        prompts=["test"] * B,
        action_mask=action_mask,
    )

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
    log(f"Forward pass: loss={loss.item():.4f}")

    # Sample pass
    model.eval()
    with torch.no_grad():
        sampled = model.sample(
            video=video,
            joint_state=joint_state,
            prompts=["test"] * B,
            steps=4,
        )
    assert sampled.shape == (B, 4, 32), f"Expected (2, 4, 32), got {sampled.shape}"
    log(f"Sample pass: shape={sampled.shape}")


def test_pretrained_load():
    """Test loading real Cosmos 2.0 pretrained weights (component-by-component)."""
    print("\n=== Test 2: Cosmos2 Pretrained Weight Loading ===")
    from mimic_video.cosmos_predict import Cosmos2PredictWrapper

    device = torch.device("cuda")
    wrapper = Cosmos2PredictWrapper(
        model_name="nvidia/Cosmos-Predict2-2B-Video2World",
        extract_layer=19,
    )

    log(f"Transformer loaded: {wrapper.transformer.config.num_layers} layers, "
        f"heads={wrapper.transformer.config.num_attention_heads}, "
        f"dim={wrapper.dim_latent}")
    log(f"VAE loaded: z_dim={wrapper.vae_latent_channels}, "
        f"temporal_compress={wrapper.vae_temporal_compression_ratio}, "
        f"spatial_compress={wrapper.vae_spatial_compression_ratio}")
    log(f"Text encoder loaded: d_model={wrapper.text_encoder.config.d_model}")

    # Move to GPU with bf16
    wrapper = wrapper.to(device=device, dtype=torch.bfloat16)
    wrapper.eval()

    # Quick forward pass
    video = torch.randn(1, 5, 3, 480, 480, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = wrapper(video, prompts=["a robot arm picks up a cube"])

    log(f"Pretrained forward: output shape={out.shape}")

    # Check dim_latent matches expected
    expected_dim = 16 * 128  # num_attention_heads * attention_head_dim = 2048
    assert wrapper.dim_latent == expected_dim, f"Expected dim_latent={expected_dim}, got {wrapper.dim_latent}"
    log(f"dim_latent={wrapper.dim_latent} matches expected 2048")

    del wrapper
    torch.cuda.empty_cache()


def test_training_loop():
    """Test training loop with tiny Cosmos 2.0 + bf16."""
    print("\n=== Test 3: Cosmos2 Training Loop (5 steps) ===")
    from mimic_video.cosmos_predict import Cosmos2PredictWrapper
    from mimic_video.mimic_video import MimicVideo

    device = torch.device("cuda")
    wrapper = Cosmos2PredictWrapper(random_weights=True, tiny=True, extract_layer=0)

    model = MimicVideo(
        dim=32,
        video_predict_wrapper=wrapper,
        action_chunk_len=4,
        dim_action=32,
        dim_joint_state=64,
        depth=1,
        dim_head=16,
        heads=2,
        num_video_viewpoints=1,
    ).to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(5):
        B = 2
        video = torch.randn(B, 1, 1, 3, 64, 64, device=device)
        actions = torch.randn(B, 4, 32, device=device)
        joint_state = torch.randn(B, 64, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(
                video=video,
                actions=actions,
                joint_state=joint_state,
                prompts=["test"] * B,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        log(f"Step {step}: loss={loss.item():.4f}")

    log(f"Training complete: {len(losses)} steps, final_loss={losses[-1]:.4f}")


def main():
    print("=" * 60)
    print("GPU Test: Cosmos 2.0 Integration")
    print("=" * 60)

    tests = [
        ("Tiny Forward", test_tiny_forward),
        ("Pretrained Load", test_pretrained_load),
        ("Training Loop", test_training_loop),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            log(f"{name} FAILED: {e}", ok=False)
            traceback.print_exc()

    passed = sum(1 for _, ok in RESULTS if ok)
    failed = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        print("\nFailed tests:")
        for msg, ok in RESULTS:
            if not ok:
                print(f"  ✗ {msg}")
        sys.exit(1)
    else:
        print("\nAll Cosmos 2.0 tests PASSED!")


if __name__ == "__main__":
    main()
