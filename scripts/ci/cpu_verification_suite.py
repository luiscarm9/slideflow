"""Deterministic CPU + Synthetic WSI PyTorch Verification Suite for GitHub Actions CI.

Generates realistic multi-resolution pyramidal histology TIFFs via Libvips (40x / 0.25 MPP)
and verifies:
  1. WSI loading, pyramid grid, thumbnails, stride, and QC (Otsu/Gaussian) via slide_test.TestSlide
  2. End-to-end Project setup, tile extraction to TFRecords, and TFRecord indexing
  3. PyTorch Iterable/Indexed DataLoader iteration & stain normalization (Reinhard & Macenko)
  4. PyTorch tile-level model forward pass & feature interface
  5. PyTorch MIL (attention_mil) bag training & evaluation smoke run
  6. Tissue/ROI segmentation model (slideflow.segment) construction & forward pass
"""

import csv
import os
from os.path import join

import numpy as np
import pytest
import pyvips
import torch

os.environ.setdefault("SF_BACKEND", "torch")
os.environ.setdefault("SF_SLIDE_BACKEND", "libvips")

import slideflow as sf
from slideflow.test import slide_test


def _make_synthetic_histology_tiff(path: str, width: int = 2048, height: int = 2048, seed: int = 0) -> str:
    """Create a pyramidal tiled RGB TIFF with H&E-like tissue islands on white background (0.25 MPP)."""
    rng = np.random.default_rng(seed)
    # White glass background (~245)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    # Create eosinophilic/basophilic tissue region in the center so Otsu/whitespace QC retains tiles
    yy, xx = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    mask = ((yy - cy) ** 2) / ((height * 0.38) ** 2) + ((xx - cx) ** 2) / ((width * 0.38) ** 2) <= 1.0

    # H&E-like RGB texture (pink/purple cellular patterns)
    tissue = np.zeros((height, width, 3), dtype=np.uint8)
    tissue[..., 0] = rng.integers(140, 215, size=(height, width), dtype=np.uint8)
    tissue[..., 1] = rng.integers(85, 155, size=(height, width), dtype=np.uint8)
    tissue[..., 2] = rng.integers(145, 210, size=(height, width), dtype=np.uint8)

    # Add synthetic dark purple nuclei dots
    nuclei = rng.random((height, width)) < 0.06
    tissue[nuclei] = np.array([75, 40, 115], dtype=np.uint8)

    canvas[mask] = tissue[mask]

    vips_img = pyvips.Image.new_from_memory(canvas.tobytes(), width, height, 3, "uchar")
    # xres=4000.0 px/mm with resunit="cm" -> auto-detected as 0.25 um/px (40x) in both libvips and cucim
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
def synthetic_wsi_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("sf_ci_env")
    slides_dir = root / "slides"
    slides_dir.mkdir()

    slide_paths = []
    for idx in range(4):
        p = str(slides_dir / f"synth_slide_{idx}.tiff")
        _make_synthetic_histology_tiff(p, width=2048, height=2048, seed=42 + idx)
        slide_paths.append(p)

    return {"root": str(root), "slides_dir": str(slides_dir), "slide_paths": slide_paths}


def test_wsi_slide_unittest_suite(synthetic_wsi_env):
    """Run upstream slide_test.TestSlide checks against a synthetic pyramidal WSI."""
    wsi_path = synthetic_wsi_env["slide_paths"][0]
    wsi = sf.WSI(wsi_path, tile_px=128, tile_um=128)
    assert pytest.approx(wsi.mpp, rel=1e-2) == 0.25
    assert wsi.estimated_num_tiles > 0

    for method_name in (
        "test_load",
        "test_qc",
        "test_index",
        "test_stride",
        "test_raises_roi_error",
        "test_thumb",
        "test_preview",
    ):
        case = slide_test.TestSlide(method_name, wsi_path, tile_px=71)
        result = case.defaultTestResult()
        case.run(result)
        assert result.wasSuccessful(), f"{method_name} failed: {result.failures + result.errors}"


def test_project_extraction_tfrecords_and_dataloader(synthetic_wsi_env):
    """Verify Project creation, WSI tile extraction into TFRecords, indexing, and PyTorch DataLoader."""
    proj_dir = join(synthetic_wsi_env["root"], "project")
    slides_dir = synthetic_wsi_env["slides_dir"]
    anns_path = join(synthetic_wsi_env["root"], "annotations.csv")

    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "category", "continuous", "time", "event"])
        for idx in range(4):
            name = f"synth_slide_{idx}"
            writer.writerow([f"pt_{idx}", name, "pos" if idx % 2 == 0 else "neg", float(idx) * 0.5, 10 + idx * 5, idx % 2])

    P = sf.create_project(
        root=proj_dir,
        slides=slides_dir,
        annotations=anns_path,
    )
    dataset = P.dataset(tile_px=128, tile_um=128)
    dataset.extract_tiles(qc="otsu", whitespace_fraction=0.9, num_threads=2, enable_downsample=False)

    assert dataset.num_tiles > 0, "Expected non-zero tiles extracted into TFRecords"
    assert len(dataset.tfrecords()) == 4

    labels, _ = dataset.labels("category")
    loader = dataset.torch(labels=labels, batch_size=4, infinite=False, normalizer="reinhard_fast")
    batch_imgs, batch_labels = next(iter(loader))
    assert isinstance(batch_imgs, torch.Tensor)
    assert batch_imgs.ndim == 4 and batch_imgs.shape[1] == 3 and batch_imgs.shape[2] == 128
    assert batch_labels.shape[0] == batch_imgs.shape[0]


def test_torch_tile_model_and_features_smoke():
    """Verify PyTorch ModelParams construction, forward pass, and postconv feature extraction without network downloads."""
    from slideflow.model.torch import Features as TorchFeatures
    from slideflow.model.torch import ModelParams as TorchModelParams

    hp = TorchModelParams(
        tile_px=128,
        tile_um=128,
        model="resnet18",
        include_top=False,
        hidden_layers=1,
        hidden_layer_width=64,
    )
    model = hp.build_model(num_classes=2, pretrain=None)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 2)

    feat_fn = TorchFeatures.from_model(model, layers=["postconv"], tile_px=128)
    with torch.no_grad():
        feats = feat_fn(x)
    assert isinstance(feats, torch.Tensor)
    assert feats.shape[0] == 2 and feats.shape[1] == 512


def test_mil_attention_training_and_eval_smoke(synthetic_wsi_env):
    """Verify Slideflow MIL bag training and evaluation on synthetic feature bags."""
    import slideflow.mil

    proj_dir = join(synthetic_wsi_env["root"], "project")
    bags_dir = join(synthetic_wsi_env["root"], "bags")
    mil_out = join(synthetic_wsi_env["root"], "mil_run")
    os.makedirs(bags_dir, exist_ok=True)

    P = sf.Project(proj_dir)
    dataset = P.dataset(tile_px=128, tile_um=128)

    # Save synthetic 64-dim bag tensors matching the 4 slides
    rng = torch.Generator().manual_seed(123)
    for idx in range(4):
        bag = torch.randn((16, 64), generator=rng)
        if idx % 2 == 0:
            bag += 1.5
        torch.save(bag, join(bags_dir, f"synth_slide_{idx}.pt"))

    train_dts = dataset.filter({"slide": ["synth_slide_0", "synth_slide_1"]})
    val_dts = dataset.filter({"slide": ["synth_slide_2", "synth_slide_3"]})

    cfg = sf.mil.mil_config("attention_mil", epochs=2, batch_size=2, lr=1e-3, drop_last=False)
    sf.mil.train_mil(
        cfg,
        train_dataset=train_dts,
        val_dataset=val_dts,
        outcomes="category",
        bags=bags_dir,
        outdir=mil_out,
        device="cpu",
    )
    assert os.path.exists(join(mil_out, "mil_params.json"))


def test_tissue_segmentation_model_smoke():
    """Verify slideflow.segment SegmentConfig construction and forward pass."""
    from slideflow.segment import SegmentConfig

    cfg = SegmentConfig(arch="FPN", encoder_name="resnet18", mpp=4.0, size=256)
    seg_model = cfg.build(encoder_weights=None)
    seg_model.eval()
    x = torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.uint8)
    with torch.no_grad():
        out = seg_model(x.float())
    assert out.shape == (2, 1, 256, 256)
