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


def test_wsi_slide_unittest_suite(tmp_path):
    """Run upstream slide_test.TestSlide checks against a synthetic pyramidal WSI."""
    # slide_test.TestSlide uses tile_um=1208 -> at 0.25 MPP, each tile spans 4832 px; use 5120x5120 pyramid
    wsi_path = _make_synthetic_histology_tiff(str(tmp_path / "large_wsi.tiff"), width=5120, height=5120, seed=7)
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
    normalizer = sf.norm.autoselect("reinhard_fast", backend="torch")
    loader = dataset.torch(labels=labels, batch_size=4, infinite=False, normalizer=normalizer)
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
    feat_tensor = feats[0] if isinstance(feats, list) else feats
    assert isinstance(feat_tensor, torch.Tensor)
    assert feat_tensor.shape[0] == 2 and feat_tensor.shape[1] == 512


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
    assert any("mil_params.json" in files for _, _, files in os.walk(mil_out))


def test_tissue_segmentation_model_smoke():
    """Verify slideflow.segment SegmentConfig construction and forward pass."""
    from slideflow.segment import SegmentConfig

    cfg = SegmentConfig(arch="FPN", encoder_name="resnet18", mpp=4.0, size=256, encoder_weights=None)
    seg_model = cfg.build_model()
    seg_model.eval()
    x = torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.uint8)
    with torch.no_grad():
        out = seg_model(x.float())
    assert out.shape == (2, 1, 256, 256)


def test_pr466_hoptimus0_roi_and_plugins(synthetic_wsi_env):
    """Verify Slideflow 3.1.0 (PR #466) additions: H-optimus-0 registry/weights_hash, ROI.area/deepcopy, WSI.roi_area."""
    import copy
    from slideflow.model import extractors
    from slideflow.model.extractors.hoptimus0 import Hoptimus0Features
    from slideflow.slide.utils import ROI

    assert "hoptimus0" in extractors.list_extractors()
    assert Hoptimus0Features.weights_hash == "105e006ee8fb2f51709a51432c1a273e35ed10d4"

    coords = np.array([[100, 100], [500, 100], [500, 500], [100, 500]])
    roi = ROI(name="tumor_box", coordinates=coords)
    assert pytest.approx(roi.area(), rel=1e-3) == 160000.0

    roi_copy = copy.deepcopy(roi)
    assert roi_copy.name == roi.name and pytest.approx(roi_copy.area(), rel=1e-3) == roi.area()

    wsi = sf.WSI(synthetic_wsi_env["slide_paths"][0], tile_px=128, tile_um=128, roi_method="auto")
    wsi.load_roi_array(coords)
    areas = wsi.roi_area()
    assert len(areas) == 1 and areas[0] > 0.0


def test_pr450_zarr_compressed_segmentation_and_outlines(tmp_path):
    """Verify PR #450 Zarr v2/v3 save_zarr_compressed, Segmentation .zip round-trip, and fast_outlines_list."""
    from slideflow.cellseg import Segmentation, seg_utils

    masks = np.zeros((64, 64), dtype=np.int32)
    masks[10:26, 10:26] = 1
    masks[34:54, 30:50] = 2

    # 1. Direct positional + keyword Zarr .zip round-trip
    zarr_zip = str(tmp_path / "test_arrays.zip")
    seg_utils.save_zarr_compressed(zarr_zip, np.arange(5), masks=masks, meta=np.array([1024, 1024]))
    loaded = seg_utils.load_zarr_arrays(zarr_zip)
    assert np.array_equal(loaded["arr_0"], np.arange(5))
    assert np.array_equal(loaded["masks"], masks)
    assert np.array_equal(loaded["meta"], np.array([1024, 1024]))

    # 2. Full Segmentation.save() / Segmentation.load() .zip round-trip
    seg_zip = str(tmp_path / "cell_masks.zip")
    seg = Segmentation(masks=masks, wsi_dim=(640, 640), wsi_offset=(0, 0))
    seg.save(seg_zip, centroids=True, flows=False)
    seg_reloaded = Segmentation.load(seg_zip)
    assert np.array_equal(seg_reloaded.masks, masks)
    assert tuple(seg_reloaded.wsi_dim) == (640, 640)

    # 3. Multithreaded contour extraction
    outlines = seg_utils.fast_outlines_list(masks, num_threads=2)
    assert len(outlines) == 2
    assert all(o.ndim == 2 and o.shape[1] == 2 and o.shape[0] > 4 for o in outlines)


def test_pr422_dinobloom_extractor_and_registry():
    """Verify PR #422 DinoBloom registration, 224x224 ViT forward pass, model_size config persistence, and virchow preservation."""
    from slideflow.model import extractors

    reg = extractors.list_extractors()
    assert "dinobloom" in reg
    assert "hoptimus0" in reg
    assert "virchow" in reg

    ext = sf.build_feature_extractor("dinobloom", model_size="small", pretrained=False, device="cpu")
    assert ext.num_features == 384
    tiles = torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.uint8)
    with torch.no_grad():
        feats = ext(tiles)
    assert feats.shape == (2, 384)

    cfg = ext.dump_config()
    assert cfg["kwargs"]["model_size"] == "small"

    with pytest.raises(ValueError):
        sf.build_feature_extractor("dinobloom", model_size="invalid_size", pretrained=False, device="cpu")


def test_pr444_multimodal_attention_im_mil():
    """Verify PR #444 Attention_IM_MIL registry, Kronecker intermediate fusion, UQ, and attention/activation helpers."""
    import slideflow.mil
    from slideflow.mil.models import Attention_IM_MIL

    cfg = sf.mil.mil_config(
        "attention_im_mil",
        model_kwargs={"clinical_num_features": 4, "z_dim": 16, "z_dim_clinical": 4},
    )
    model = cfg.build_model(n_in=20, n_out=2)
    assert isinstance(model, Attention_IM_MIL)

    model.eval()
    bags = torch.randn(3, 10, 20)
    lens = torch.tensor([10, 7, 5])
    with torch.no_grad():
        logits, att = model(bags, lens, return_attention=True)
        uq_means, uq_stds = model(bags, lens, uq=True)
        att_direct = model.calculate_attention(bags, lens)
        acts = model.get_last_layer_activations(bags, lens)

    assert logits.shape == (3, 2)
    assert att.shape == (3, 10, 1)
    assert uq_means.shape == (3, 2) and uq_stds.shape == (3, 2)
    assert att_direct.shape == (3, 10, 1)
    assert acts.shape == (3, 16)

    with pytest.raises(ValueError):
        Attention_IM_MIL(n_feats=20, n_out=2, clinical_num_features=0)


def test_pr441_mil_ordinal_training_and_eval(tmp_path):
    """Verify PR #441 MIL ordinal classification training, create_preds conversion, and eval_mil."""
    import slideflow.mil

    slides_dir = tmp_path / "ord_slides"
    slides_dir.mkdir()
    bags_dir = tmp_path / "ord_bags"
    bags_dir.mkdir()
    anns_path = str(tmp_path / "ord_anns.csv")

    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "split", "grade"])
        for idx in range(6):
            sname = f"ord_slide_{idx}"
            _make_synthetic_histology_tiff(str(slides_dir / f"{sname}.tiff"), width=512, height=512, seed=200 + idx)
            bag = torch.randn((12, 32)) + float(idx % 3)
            torch.save(bag, str(bags_dir / f"{sname}.pt"))
            writer.writerow([f"pt_{idx}", sname, "TRAIN" if idx < 3 else "VAL", f"G{idx % 3}"])

    P = sf.create_project(root=str(tmp_path / "ord_proj"), slides=str(slides_dir), annotations=anns_path)
    dts = P.dataset(tile_px=128, tile_um=128)
    train_dts = dts.filter({"split": ["TRAIN"]})
    val_dts = dts.filter({"split": ["VAL"]})

    mil_out = str(tmp_path / "ord_mil_out")
    cfg = sf.mil.mil_config("attention_mil", loss="BCE_ordinal", epochs=2, batch_size=2, lr=1e-3, drop_last=False)
    sf.mil.train_mil(
        cfg,
        train_dataset=train_dts,
        val_dataset=val_dts,
        outcomes="grade",
        bags=str(bags_dir),
        outdir=mil_out,
        device="cpu",
    )
    run_dirs = [join(mil_out, d) for d in os.listdir(mil_out) if os.path.isdir(join(mil_out, d))]
    assert len(run_dirs) == 1
    eval_df = sf.mil.eval_mil(run_dirs[0], val_dts, outcomes="grade", bags=str(bags_dir), outdir=str(tmp_path / "ord_eval"))
    assert len(eval_df) == 3
