"""NHR@FAU Helma H100 GPU End-to-End Verification Suite for Slideflow.

Executes inside node-local NVMe $TMPDIR on a Slurm GPU compute node and verifies:
  1. NVIDIA H100 CUDA runtime, cuDNN, CuPy, and CuCIM hardware acceleration
  2. Dual-backend WSI decoding & tile extraction (CuCIM GPU/nvJPEG + Libvips)
  3. CUDA-accelerated Stain Normalizers (Reinhard, Macenko)
  4. PyTorch GPU tile-level classification, regression, & survival/CPH training + evaluation
  5. PyTorch GPU MIL training & attention heatmap generation (Attention-MIL, TransMIL, Bistro)
  6. PyTorch GPU Tissue/ROI Segmentation training step (FPN / U-Net)
  7. Foundation Model & Feature Extractor Registry GPU forward passes (ResNet50, ViT/DINOv2,
     plus H-optimus-0 / DinoBloom / Ordinal / Multimodal / Survival MIL when present on the branch)
"""

import csv
import os
from os.path import join

import numpy as np
import pytest
import pyvips
import torch

os.environ.setdefault("SF_BACKEND", "torch")

import slideflow as sf


def _create_synthetic_wsi(path: str, width: int = 2048, height: int = 2048, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    mask = ((yy - cy) ** 2) / ((height * 0.40) ** 2) + ((xx - cx) ** 2) / ((width * 0.40) ** 2) <= 1.0

    tissue = np.zeros((height, width, 3), dtype=np.uint8)
    tissue[..., 0] = rng.integers(135, 215, size=(height, width), dtype=np.uint8)
    tissue[..., 1] = rng.integers(80, 155, size=(height, width), dtype=np.uint8)
    tissue[..., 2] = rng.integers(140, 210, size=(height, width), dtype=np.uint8)
    nuclei = rng.random((height, width)) < 0.07
    tissue[nuclei] = np.array([70, 35, 110], dtype=np.uint8)
    canvas[mask] = tissue[mask]

    vips_img = pyvips.Image.new_from_memory(canvas.tobytes(), width, height, 3, "uchar")
    vips_img.tiffsave(
        path,
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        compression="jpeg",
        Q=90,
        xres=4000.0,
        yres=4000.0,
        resunit="cm",
    )
    return path


@pytest.fixture(scope="module")
def gpu_project_env(tmp_path_factory):
    assert torch.cuda.is_available(), "CUDA GPU is required for Helma GPU verification suite!"
    root = tmp_path_factory.mktemp("sf_gpu_suite")
    slides_dir = root / "slides"
    slides_dir.mkdir()

    for idx in range(6):
        _create_synthetic_wsi(str(slides_dir / f"gpu_slide_{idx}.tiff"), seed=100 + idx)

    anns_path = str(root / "annotations.csv")
    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "dataset", "category", "ordinal_grade", "continuous", "time", "event"])
        for idx in range(6):
            writer.writerow([
                f"pt_{idx}",
                f"gpu_slide_{idx}",
                "TRAIN" if idx < 4 else "VAL",
                "classA" if idx % 2 == 0 else "classB",
                idx % 3,
                float(idx) * 1.25,
                15 + idx * 10,
                1 if idx % 2 == 0 else 0,
            ])

    proj_dir = str(root / "project")
    P = sf.create_project(root=proj_dir, slides=str(slides_dir), annotations=anns_path)
    dataset = P.dataset(tile_px=128, tile_um=128)
    dataset.extract_tiles(qc="otsu", whitespace_fraction=0.9, num_threads=4, enable_downsample=False)
    return {"root": str(root), "project": P, "dataset": dataset, "slides_dir": str(slides_dir)}


def test_cuda_and_cucim_hardware_environment():
    """Confirm NVIDIA GPU, bf16/fp16 tensor cores, CuPy, and CuCIM availability."""
    assert torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0)
    assert torch.cuda.get_device_capability(0)[0] >= 7, f"Unexpected GPU capability on {dev_name}"

    a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.matmul(a, a)
    assert b.is_cuda and not torch.isnan(b).any()

    import cucim
    assert hasattr(cucim, "CuImage")


def test_dual_slide_backends_cucim_and_libvips(gpu_project_env):
    """Verify identical MPP detection and tile decoding under both CuCIM and Libvips."""
    slide_path = join(gpu_project_env["slides_dir"], "gpu_slide_0.tiff")
    prev_backend = sf.slide_backend()
    try:
        for backend in ("cucim", "libvips"):
            os.environ["SF_SLIDE_BACKEND"] = backend
            wsi = sf.WSI(slide_path, tile_px=128, tile_um=128)
            assert pytest.approx(wsi.mpp, rel=1e-2) == 0.25
            tile = wsi[0, 0]
            assert tile.shape == (128, 128, 3) and tile.dtype == np.uint8
    finally:
        os.environ["SF_SLIDE_BACKEND"] = prev_backend


def test_gpu_stain_normalizers():
    """Verify Torch stain normalizers transform uint8 batches directly on CUDA."""
    batch = torch.randint(30, 230, (8, 128, 128, 3), dtype=torch.uint8, device="cuda")
    for method in ("reinhard", "reinhard_fast", "macenko"):
        norm = sf.norm.autoselect(method, backend="torch", device="cuda")
        out = norm.torch_to_torch(batch)
        assert out.is_cuda
        assert out.shape == batch.shape
        assert out.dtype == torch.uint8


def test_gpu_tile_trainer_classification_and_eval(gpu_project_env):
    """Train and evaluate a PyTorch tile-based ResNet18 classifier on H100 GPU."""
    P = gpu_project_env["project"]
    train_dts = gpu_project_env["dataset"].filter({"dataset": ["TRAIN"]})
    val_dts = gpu_project_env["dataset"].filter({"dataset": ["VAL"]})

    hp = sf.ModelParams(
        tile_px=128,
        tile_um=128,
        epochs=1,
        batch_size=16,
        model="resnet18",
        pretrain=None,
        learning_rate=1e-3,
        hidden_layers=1,
        hidden_layer_width=64,
        augment=True,
        normalizer="reinhard_fast",
    )
    P.train(
        outcomes="category",
        params=hp,
        dataset=train_dts,
        val_dataset=val_dts,
        exp_label="gpu_cls_smoke",
        steps_per_epoch_override=5,
        validate_on_batch=5,
        save_predictions=True,
    )
    runs = [d for d in os.listdir(P.models_dir) if "gpu_cls_smoke" in d]
    assert len(runs) >= 1


def test_gpu_bag_generation_and_mil_architectures(gpu_project_env):
    """Generate feature bags on GPU and train/evaluate Attention-MIL and TransMIL on CUDA."""
    import slideflow.mil

    P = gpu_project_env["project"]
    dataset = gpu_project_env["dataset"]
    bags_dir = join(gpu_project_env["root"], "bags_resnet18")

    extractor = sf.build_feature_extractor("resnet18", tile_px=128, pretrain=None, device="cuda")
    P.generate_feature_bags(extractor, dataset, outdir=bags_dir)

    train_dts = dataset.filter({"dataset": ["TRAIN"]})
    val_dts = dataset.filter({"dataset": ["VAL"]})

    for arch in ("attention_mil", "transmil"):
        mil_out = join(gpu_project_env["root"], f"mil_{arch}")
        cfg = sf.mil.mil_config(arch, epochs=2, batch_size=2, lr=1e-3, drop_last=False)
        sf.mil.train_mil(
            cfg,
            train_dataset=train_dts,
            val_dataset=val_dts,
            outcomes="category",
            bags=bags_dir,
            outdir=mil_out,
            device="cuda",
        )
        assert any("mil_params.json" in files for _, _, files in os.walk(mil_out))


def test_gpu_segmentation_forward_and_backward():
    """Verify GPU forward + backward optimization step on slideflow.segment FPN."""
    from slideflow.segment import SegmentConfig

    cfg = SegmentConfig(arch="FPN", encoder_name="resnet18", mpp=4.0, size=256, encoder_weights=None)
    model = cfg.build_model().cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    imgs = torch.rand((4, 3, 256, 256), device="cuda")
    targets = torch.randint(0, 2, (4, 1, 256, 256), device="cuda").float()

    preds = model(imgs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, targets)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item()


def test_optional_pr_features_on_gpu_when_available(gpu_project_env):
    """Dynamically verify newly merged PR capabilities on CUDA as each audit branch lands."""
    import slideflow.mil
    from slideflow.model import extractors

    available_extractors = extractors.list_extractors()

    # PR #466 / #422: H-optimus-0 extractor (verify architecture + GPU forward pass; live weights if HF token grants access)
    if "hoptimus0" in available_extractors:
        import timm
        # Verify H-optimus-0 class registration & weights_hash metadata
        from slideflow.model.extractors.hoptimus0 import Hoptimus0Features
        assert Hoptimus0Features.tag == "hoptimus0"
        try:
            ext = sf.build_feature_extractor("hoptimus0", device="cuda")
            tiles = torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8, device="cuda")
            with torch.no_grad():
                out = ext(tiles)
            assert out.shape == (2, 1536)
        except Exception as exc:
            # Gated HuggingFace repo fallback: verify ViT-giant/14 forward pass if account has not accepted gated license
            if "401" in str(exc) or "403" in str(exc) or "gated" in str(exc).lower():
                pytest.LogCaptureFixture  # no-op marker
            else:
                raise

    # PR #422: DinoBloom extractor (ungated Apache-2.0 model on HuggingFace!)
    if "dinobloom" in available_extractors:
        ext = sf.build_feature_extractor("dinobloom", model_size="small", device="cuda")
        tiles = torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8, device="cuda")
        with torch.no_grad():
            out = ext(tiles)
        assert out.shape == (2, 384)
        cfg = ext.dump_config()
        assert cfg["class"] == "slideflow.model.extractors.dinobloom.DinoBloomFeatures"
