"""
Smoke test for LKFormer development environment.
Verifies that core model architectures can be instantiated and run forward passes.
"""
import sys
import torch
import numpy as np

sys.path.insert(0, '.')


def test_lkformer():
    print("=" * 60)
    print("Testing LKFormer (Restormer) - Paper's main contribution")
    print("=" * 60)

    from SRModel.LKFormer import Restormer

    for upscale in [2, 4]:
        model = Restormer(
            inp_channels=3, out_channels=3, dim=48,
            upscale=upscale, num_blocks=[2, 2, 2, 2], heads=[1, 2, 4, 8]
        )
        model.eval()

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)

        expected_h = 32 * upscale
        expected_w = 32 * upscale
        assert out.shape == (1, 3, expected_h, expected_w), \
            f"Expected {(1, 3, expected_h, expected_w)}, got {out.shape}"

        params = sum(p.numel() for p in model.parameters())
        print(f"  x{upscale} upscale: input {tuple(x.shape)} -> output {tuple(out.shape)} "
              f"({params:,} params) [OK]")

    print()


def test_swinir():
    print("=" * 60)
    print("Testing SwinIR - Baseline model")
    print("=" * 60)

    from models.network_swinir import SwinIR

    model = SwinIR(
        upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.,
        depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    )
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 128, 128), f"Expected (1,3,128,128), got {out.shape}"
    params = sum(p.numel() for p in model.parameters())
    print(f"  x2 upscale: input {tuple(x.shape)} -> output {tuple(out.shape)} "
          f"({params:,} params) [OK]")
    print()


def test_image_processing():
    print("=" * 60)
    print("Testing image processing pipeline (numpy + cv2)")
    print("=" * 60)

    import cv2

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    assert img_resized.shape == (128, 128, 3)
    print(f"  Bicubic resize: {img.shape} -> {img_resized.shape} [OK]")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img_gray.shape == (64, 64)
    print(f"  Color conversion: {img.shape} -> {img_gray.shape} [OK]")
    print()


if __name__ == '__main__':
    print("\nLKFormer Development Environment Smoke Test")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    test_lkformer()
    test_swinir()
    test_image_processing()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
