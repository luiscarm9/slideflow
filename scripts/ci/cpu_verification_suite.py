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
from os.path import exists, join

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
        for idx in range(8):
            sname = f"ord_slide_{idx}"
            _make_synthetic_histology_tiff(str(slides_dir / f"{sname}.tiff"), width=512, height=512, seed=200 + idx)
            bag = torch.randn((12, 32)) + float(idx % 3)
            torch.save(bag, str(bags_dir / f"{sname}.pt"))
            writer.writerow([f"pt_{idx}", sname, "TRAIN" if idx < 4 else "VAL", f"G{idx % 3}"])

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
    assert len(eval_df) == 4


def test_pr447_mil_survival_training_and_eval(tmp_path):
    """Verify PR #447 MIL Cox Proportional Hazards survival training, concordance index, and eval_mil."""
    import slideflow.mil

    slides_dir = tmp_path / "surv_slides"
    slides_dir.mkdir()
    bags_dir = tmp_path / "surv_bags"
    bags_dir.mkdir()
    anns_path = str(tmp_path / "surv_anns.csv")

    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "split", "time", "event"])
        for idx in range(8):
            sname = f"surv_slide_{idx}"
            _make_synthetic_histology_tiff(str(slides_dir / f"{sname}.tiff"), width=512, height=512, seed=300 + idx)
            bag = torch.randn((12, 32)) + float(idx) * 0.25
            torch.save(bag, str(bags_dir / f"{sname}.pt"))
            writer.writerow([f"pt_{idx}", sname, "TRAIN" if idx < 4 else "VAL", 10 + idx * 8, 1 if idx % 2 == 0 else 0])

    P = sf.create_project(root=str(tmp_path / "surv_proj"), slides=str(slides_dir), annotations=anns_path)
    dts = P.dataset(tile_px=128, tile_um=128)
    train_dts = dts.filter({"split": ["TRAIN"]})
    val_dts = dts.filter({"split": ["VAL"]})

    mil_out = str(tmp_path / "surv_mil_out")
    cfg = sf.mil.mil_config("attention_mil", loss="CPH", epochs=2, batch_size=2, lr=1e-3, drop_last=False)
    sf.mil.train_mil(
        cfg,
        train_dataset=train_dts,
        val_dataset=val_dts,
        outcomes="time",
        events="event",
        bags=str(bags_dir),
        outdir=mil_out,
        device="cpu",
    )
    run_dirs = [join(mil_out, d) for d in os.listdir(mil_out) if os.path.isdir(join(mil_out, d))]
    assert len(run_dirs) == 1
    eval_df = sf.mil.eval_mil(
        run_dirs[0],
        val_dts,
        outcomes="time",
        events="event",
        bags=str(bags_dir),
        outdir=str(tmp_path / "surv_eval"),
    )
    assert len(eval_df) == 4


def test_pr410_balanced_bags_patient_mil(tmp_path):
    """Verify PR #410 balanced bag sampling across multi-slide patients and patient-level MIL training."""
    import slideflow.mil
    from slideflow.mil.data import BagDataset

    # Verify equal per-slide sampling: slide A has 100 ones, slide B has 10 twos; bag_size=8 -> 4 from A, 4 from B
    bag_a = torch.ones((100, 8), dtype=torch.float32)
    bag_b = torch.full((10, 8), 2.0, dtype=torch.float32)
    ds = BagDataset([[bag_a, bag_b]], bag_size=8, balanced=True)
    sampled, length = ds[0]
    assert length == 8 and sampled.shape == (8, 8)
    assert int((sampled[:, 0] == 1.0).sum()) == 4
    assert int((sampled[:, 0] == 2.0).sum()) == 4

    with pytest.raises(TypeError):
        BagDataset([bag_a], bag_size=8, balanced=True)

    # End-to-end patient-level MIL training with balanced=True
    slides_dir = tmp_path / "bal_slides"
    slides_dir.mkdir()
    bags_dir = tmp_path / "bal_bags"
    bags_dir.mkdir()
    anns_path = str(tmp_path / "bal_anns.csv")

    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "split", "category"])
        for pt in range(8):
            for sub in range(2):
                sname = f"bal_pt{pt}_s{sub}"
                _make_synthetic_histology_tiff(str(slides_dir / f"{sname}.tiff"), width=512, height=512, seed=400 + pt * 2 + sub)
                torch.save(torch.randn((10 + sub * 10, 16)), str(bags_dir / f"{sname}.pt"))
                writer.writerow([f"pt_{pt}", sname, "TRAIN" if pt < 4 else "VAL", "pos" if pt % 2 == 0 else "neg"])

    P = sf.create_project(root=str(tmp_path / "bal_proj"), slides=str(slides_dir), annotations=anns_path)
    dts = P.dataset(tile_px=128, tile_um=128)
    cfg = sf.mil.mil_config(
        "attention_mil",
        aggregation_level="patient",
        balanced=True,
        bag_size=8,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        drop_last=False,
    )
    sf.mil.train_mil(
        cfg,
        train_dataset=dts.filter({"split": ["TRAIN"]}),
        val_dataset=dts.filter({"split": ["VAL"]}),
        outcomes="category",
        bags=str(bags_dir),
        outdir=str(tmp_path / "bal_mil_out"),
        device="cpu",
    )


def test_pr408_torch_cph_tile_survival_trainer(tmp_path):
    """Verify PR #408 PyTorch CoxProportionalHazardsLoss, SurvivalTrainer, C-index sign, evaluate, and predict."""
    from slideflow.model.torch_utils import CoxProportionalHazardsLoss

    loss_fn = CoxProportionalHazardsLoss()

    # Short survival (T=5, E=1) with higher log-hazard (+2.0) vs long survival (T=50, E=1) with lower log-hazard (-2.0)
    y_true = torch.tensor([[5.0, 1.0], [50.0, 1.0]], dtype=torch.float32)
    good_log_h = torch.tensor([[2.0], [-2.0]], dtype=torch.float32, requires_grad=True)
    bad_log_h = torch.tensor([[-2.0], [2.0]], dtype=torch.float32, requires_grad=True)
    assert loss_fn(good_log_h, y_true).item() < loss_fn(bad_log_h, y_true).item()

    # Zero-event censored batch must preserve autograd graph
    zero_event_y = torch.tensor([[10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
    pred_zero = torch.tensor([[0.5], [-0.5]], dtype=torch.float32, requires_grad=True)
    zero_loss = loss_fn(pred_zero, zero_event_y)
    assert zero_loss.item() == 0.0 and zero_loss.requires_grad
    zero_loss.backward()

    # End-to-end tile-level PyTorch SurvivalTrainer (image-only + multi-input + evaluate + predict + load)
    slides_dir = tmp_path / "cph_slides"
    slides_dir.mkdir()
    anns_path = str(tmp_path / "cph_anns.csv")

    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "split", "time", "event", "stage_covar"])
        for idx in range(6):
            sname = f"cph_slide_{idx}"
            _make_synthetic_histology_tiff(str(slides_dir / f"{sname}.tiff"), width=512, height=512, seed=500 + idx)
            writer.writerow([
                f"pt_{idx}",
                sname,
                "TRAIN" if idx < 4 else "VAL",
                10.0 + idx * 12.0,
                1 if idx % 2 == 0 else 0,
                "high" if idx < 3 else "low",
            ])

    P = sf.create_project(root=str(tmp_path / "cph_proj"), slides=str(slides_dir), annotations=anns_path)
    P.extract_tiles(tile_px=128, tile_um=128, qc=None, roi_method="ignore", num_threads=2)

    hp = sf.ModelParams(
        tile_px=128,
        tile_um=128,
        epochs=1,
        toplayer_epochs=0,
        model="resnet18",
        batch_size=4,
        learning_rate=1e-4,
        loss="CoxProportionalHazardsLoss",
        augment=False,
    )
    assert hp.model_type() == "survival"
    assert sf.ModelParams(tile_px=128, tile_um=128, model="resnet18", loss="NLL").model_type() == "survival"

    train_dts = P.dataset(tile_px=128, tile_um=128).filter({"split": ["TRAIN"]})
    val_dts = P.dataset(tile_px=128, tile_um=128).filter({"split": ["VAL"]})

    # 1. Image-only survival training (event as first input_header)
    res_single = P.train(
        outcomes="time",
        input_header="event",
        params=hp,
        dataset=train_dts,
        val_dataset=val_dts,
        exp_label="cph_single",
        pretrain=None,
        steps_per_epoch_override=2,
        use_tensorboard=False,
        mixed_precision=False,
    )
    assert len(res_single) == 1
    cph_dirs = [join(P.models_dir, d) for d in os.listdir(P.models_dir) if "cph_single" in d]
    assert len(cph_dirs) == 1
    model_dir = cph_dirs[0]
    saved_zip = [
        join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".zip")
    ][0]

    # Verify sf.model.load, P.evaluate, and prospective P.predict (without input_header)
    loaded_model = sf.model.load(saved_zip)
    assert loaded_model is not None

    eval_metrics = P.evaluate(
        saved_zip,
        outcomes="time",
        input_header="event",
        dataset=val_dts,
        batch_size=4,
        mixed_precision=False,
    )
    assert "eval" in eval_metrics and "slide_c_index" in eval_metrics["eval"]

    preds = P.predict(saved_zip, dataset=val_dts, batch_size=4, mixed_precision=False)
    assert "slide" in preds and "time-y_pred" in preds["slide"].columns
    assert not np.isnan(preds["slide"]["time-y_pred"].to_numpy()).any()

    # 2. Multi-input survival training (event + clinical covariate)
    res_multi = P.train(
        outcomes="time",
        input_header=["event", "stage_covar"],
        params=hp,
        dataset=train_dts,
        val_dataset=val_dts,
        exp_label="cph_multi",
        pretrain=None,
        steps_per_epoch_override=2,
        use_tensorboard=False,
        mixed_precision=False,
    )
    assert len(res_multi) == 1


def test_pr463_dual_magnification_extraction_and_features(tmp_path):
    """Verify PR #463 extract_lower_mag_tiles_from_tfr, LowerMagSlideReport, and DualMagnificationFeatures."""
    slides_dir = tmp_path / "dual_slides"
    slides_dir.mkdir()
    anns_path = str(tmp_path / "dual_anns.csv")

    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "category"])
        for idx in range(2):
            sname = f"dual_slide_{idx}"
            # 1024x1024 at 0.25 mpp -> 256um x 256um -> 4x4=16 tiles at 64um, combined (mag_ratio=2) -> 2x2=4 tiles at 128um
            _make_synthetic_histology_tiff(str(slides_dir / f"{sname}.tiff"), width=1024, height=1024, seed=600 + idx)
            writer.writerow([f"pt_{idx}", sname, "pos" if idx == 0 else "neg"])

    P = sf.create_project(root=str(tmp_path / "dual_proj"), slides=str(slides_dir), annotations=anns_path)
    P.extract_tiles(
        tile_px=128,
        tile_um=64,
        qc=None,
        roi_method="ignore",
        whitespace_fraction=1.0,
        grayspace_fraction=1.0,
        num_threads=2,
    )

    high_dts = P.dataset(tile_px=128, tile_um=64)
    assert len(high_dts.tfrecords()) == 2
    assert high_dts.num_tiles == 32  # 16 tiles per slide

    reports = P.extract_lower_mag_tiles_from_tfr(
        tile_px=128,
        tile_um=64,
        mag_ratio=2,
        target_tile_px=128,
        report=True,
    )
    assert len(reports) == 2
    for rep in reports.values():
        assert rep.combination_efficiency == 100.0
        assert rep.spatial_coverage_reduction == 75.0

    low_dts = P.dataset(tile_px=128, tile_um=128, source_um=64)
    assert len(low_dts.tfrecords()) == 2
    assert low_dts.num_tiles == 8  # 4 combined tiles per slide

    # Verify DualMagnificationFeatures spatial matching & feature concatenation (512 + 512 = 1024 dims)
    dual_out = str(tmp_path / "dual_features_out")
    dual_feats = P.extract_dual_magnification_features(
        model_path="resnet18_imagenet",
        output_dir=dual_out,
        device="cpu",
        high_mag_um=64,
        low_mag_um=128,
        tile_px=128,
        batch_size=4,
        num_workers=0,
    )
    assert dual_feats.num_features == 1024
    for idx in range(2):
        pt_bag = torch.load(join(dual_feats.concatenated_dir, f"dual_slide_{idx}.pt"))
        assert pt_bag.shape == (16, 1024)
        assert torch.isfinite(pt_bag).all().item()


def test_patch_311_mil_eval_heatmaps_and_dir_bags(tmp_path):
    """Regression test for upstream issues #456, #428, #457, and #431."""
    import glob
    import slideflow as sf
    from slideflow.mil import eval_mil, generate_attention_heatmaps, mil_config, train_mil

    slides_dir = tmp_path / "mil_hm_slides"
    slides_dir.mkdir()
    anns_path = str(tmp_path / "mil_hm_anns.csv")
    bags_dir = tmp_path / "mil_hm_bags"
    bags_dir.mkdir()

    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "category"])
        for idx in range(4):
            sname = f"hm_slide_{idx}"
            _make_synthetic_histology_tiff(
                str(slides_dir / f"{sname}.tiff"), width=256, height=256, seed=700 + idx
            )
            writer.writerow([f"pt_{idx}", sname, "pos" if idx % 2 == 0 else "neg"])

    P = sf.create_project(
        root=str(tmp_path / "mil_hm_proj"),
        slides=str(slides_dir),
        annotations=anns_path,
    )
    P.extract_tiles(
        tile_px=64,
        tile_um=32,
        qc=None,
        roi_method="ignore",
        whitespace_fraction=1.0,
        grayspace_fraction=1.0,
        num_threads=2,
    )
    dts = P.dataset(tile_px=64, tile_um=32)
    rng = np.random.default_rng(311)
    for slide in dts.slides():
        # 4x4 grid of 128px-step tile centers in a 256x256 slide (at 0.25 mpp -> 32um = 128px)
        locs = np.array([[64, 64], [192, 64], [64, 192], [192, 192]], dtype=np.int64)
        feats = torch.tensor(rng.normal(size=(4, 16)), dtype=torch.float32)
        torch.save(feats, str(bags_dir / f"{slide}.pt"))
        np.savez(str(bags_dir / f"{slide}.index.npz"), locs)

    # Write minimal bags_config.json matching dataset tile size
    sf.util.write_json(
        {"tile_px": 64, "tile_um": 32, "extractor": {"class": "stub", "kwargs": {}}},
        str(bags_dir / "bags_config.json"),
    )

    cfg = mil_config(
        "attention_mil",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        bag_size=4,
        fit_one_cycle=False,
    )
    train_out = str(tmp_path / "mil_hm_train")
    train_mil(
        cfg,
        train_dataset=dts,
        val_dataset=dts,
        outcomes="category",
        bags=str(bags_dir),
        outdir=train_out,
        attention_heatmaps=False,
        device="cpu",
    )
    trained_dirs = [
        join(train_out, d)
        for d in os.listdir(train_out)
        if os.path.isdir(join(train_out, d))
    ]
    assert len(trained_dirs) == 1

    eval_out = str(tmp_path / "mil_hm_eval")
    df = eval_mil(
        weights=trained_dirs[0],
        dataset=dts,
        outcomes="category",
        bags=str(bags_dir),  # directory string (#457)
        outdir=eval_out,
        attention_heatmaps=True,  # unimodal model (#456 / #428)
    )
    assert len(df) == 4
    model_dirs = glob.glob(join(eval_out, "*"))
    assert len(model_dirs) == 1
    hm_pngs = sorted(glob.glob(join(model_dirs[0], "heatmaps", "*_attn.png")))
    assert len(hm_pngs) == 4, f"Expected 4 attention heatmaps from eval_mil, got {hm_pngs}"

    # Verify #431: generate_attention_heatmaps accepts bag dir + attention dir
    att_dir = join(model_dirs[0], "attention")
    hm_from_dir = str(tmp_path / "hm_from_dir")
    generate_attention_heatmaps(
        outdir=hm_from_dir,
        dataset=dts,
        bags=str(bags_dir),
        attention=att_dir,
    )
    assert len(glob.glob(join(hm_from_dir, "*_attn.png"))) == 4

    # Verify #431: generate_attention_heatmaps accepts list of .npz attention file paths
    bag_list = sorted(dts.get_bags(str(bags_dir)))
    att_files = [join(att_dir, f"{sf.util.path_to_name(b)}_att.npz") for b in bag_list]
    hm_from_files = str(tmp_path / "hm_from_files")
    generate_attention_heatmaps(
        outdir=hm_from_files,
        dataset=dts,
        bags=bag_list,
        attention=att_files,
    )
    assert len(glob.glob(join(hm_from_files, "*_attn.png"))) == 4


def test_patch_311_virchow_and_vahadane_compat(monkeypatch):
    """Regression test for upstream issues #427 (Virchow HF default) and #451 (SPAMS fallback)."""
    import timm
    import slideflow as sf
    from slideflow.norm import StainNormalizer

    class _StubVirchow(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return torch.zeros((b, 257, 1280), dtype=torch.float32, device=x.device)

    calls = []

    def _fake_create_model(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return _StubVirchow()

    monkeypatch.setattr(timm, "create_model", _fake_create_model)
    ext = sf.build_feature_extractor("virchow", device="cpu")
    assert len(calls) == 1
    assert calls[0][0] == "hf-hub:paige-ai/Virchow"
    assert calls[0][1].get("pretrained") is True
    assert ext.num_features == 2560
    sample = torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8)
    out = ext(sample)
    assert out.shape == (2, 2560)

    # Verify actionable error when HF download fails
    def _failing_create_model(*args, **kwargs):
        raise OSError("401 Unauthorized")

    monkeypatch.setattr(timm, "create_model", _failing_create_model)
    with pytest.raises(RuntimeError, match="huggingface-cli login"):
        sf.build_feature_extractor("virchow", device="cpu")

    # Verify #451: vahadane defaults to sklearn and vahadane_spams gracefully falls back without spams
    rng = np.random.default_rng(451)
    # H&E-like synthetic tissue patch (non-white pink/purple)
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:, :, 0] = rng.integers(140, 210, size=(128, 128), dtype=np.uint8)
    img[:, :, 1] = rng.integers(80, 150, size=(128, 128), dtype=np.uint8)
    img[:, :, 2] = rng.integers(130, 200, size=(128, 128), dtype=np.uint8)

    norm_default = StainNormalizer("vahadane")
    assert norm_default.n.preset_tag == "vahadane_sklearn"
    out_default = norm_default.rgb_to_rgb(img)
    assert out_default.shape == img.shape and out_default.dtype == np.uint8

    norm_spams = StainNormalizer("vahadane_spams")
    out_spams = norm_spams.rgb_to_rgb(img)
    assert out_spams.shape == img.shape and out_spams.dtype == np.uint8


def test_patch_311_generate_feature_bags_min_tiles_and_auto_dataset(tmp_path):
    """Regression test for upstream issue #435 (min_tiles default & auto_dataset_allow_none)."""
    import slideflow as sf
    from slideflow import project_utils

    slides_dir = tmp_path / "mt_slides"
    slides_dir.mkdir()
    anns_path = str(tmp_path / "mt_anns.csv")
    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "category"])
        _make_synthetic_histology_tiff(
            str(slides_dir / "small_slide.tiff"), width=256, height=256, seed=435
        )
        writer.writerow(["pt_0", "small_slide", "pos"])

    P = sf.create_project(
        root=str(tmp_path / "mt_proj"),
        slides=str(slides_dir),
        annotations=anns_path,
    )
    # 256x256 at 0.25 mpp -> 4 tiles (< 16!)
    P.extract_tiles(
        tile_px=64,
        tile_um=32,
        qc=None,
        roi_method="ignore",
        whitespace_fraction=1.0,
        grayspace_fraction=1.0,
        num_threads=1,
    )
    dts = P.dataset(tile_px=64, tile_um=32)
    assert dts.num_tiles == 4

    ext = sf.build_feature_extractor("resnet18_imagenet", tile_px=64)
    out_bags = P.generate_feature_bags(
        ext,
        dataset=dts,
        outdir=str(tmp_path / "mt_bags"),
        batch_size=2,
        num_workers=0,
    )
    bag_pt = join(out_bags, "small_slide.pt")
    assert exists(bag_pt), "Slide with <16 tiles should not be silently dropped by default"
    assert torch.load(bag_pt).shape[0] == 4

    # Verify _filters_to_dataset allows filter args when dataset=None is forwarded
    called = {}

    class _DummyProj:
        def dataset(self, **kw):
            called["dataset_kw"] = kw
            return "auto_ds"

    def _dummy_method(obj, model, *, dataset=None, **kw):
        called["method"] = (model, dataset, kw)
        return "ok"

    wrapped = project_utils.auto_dataset_allow_none(_dummy_method)
    orig_get_cfg = sf.util.get_model_config
    try:
        sf.util.get_model_config = lambda m: {"hp": {"tile_px": 64, "tile_um": 32}}
        res = wrapped(_DummyProj(), "fake_model", min_tiles=2)
        assert res == "ok"
        assert called["dataset_kw"]["min_tiles"] == 2
        assert called["method"][1] == "auto_ds"
    finally:
        sf.util.get_model_config = orig_get_cfg


def test_patch_311_custom_model_and_loss_eval_load(tmp_path):
    """Regression test for upstream issue #449 (custom model & custom loss with evaluate/load)."""
    import glob
    import slideflow as sf
    from slideflow.model import ModelParams

    class _TinyCustomNet(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(3, num_classes)

        def forward(self, x):
            return self.fc(self.pool(x).flatten(1))

    class _TinyCustomLoss(torch.nn.CrossEntropyLoss):
        pass

    slides_dir = tmp_path / "cm_slides"
    slides_dir.mkdir()
    anns_path = str(tmp_path / "cm_anns.csv")
    with open(anns_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "slide", "category"])
        for idx in range(2):
            sname = f"cm_slide_{idx}"
            _make_synthetic_histology_tiff(
                str(slides_dir / f"{sname}.tiff"), width=256, height=256, seed=449 + idx
            )
            writer.writerow([f"pt_{idx}", sname, "pos" if idx == 0 else "neg"])

    P = sf.create_project(
        root=str(tmp_path / "cm_proj"),
        slides=str(slides_dir),
        annotations=anns_path,
    )
    P.extract_tiles(
        tile_px=64,
        tile_um=32,
        qc=None,
        roi_method="ignore",
        whitespace_fraction=1.0,
        grayspace_fraction=1.0,
        num_threads=1,
    )
    dts = P.dataset(tile_px=64, tile_um=32)

    hp = ModelParams(
        tile_px=64,
        tile_um=32,
        model=_TinyCustomNet,
        loss={"type": "classification", "fn": _TinyCustomLoss},
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        include_top=False,
        augment=False,
    )
    # Verify class-level ModelDict/AllLossDict were NOT polluted
    assert "custom" not in ModelParams.ModelDict
    assert "custom_classification" not in ModelParams.AllLossDict

    P.train(
        "category",
        params=hp,
        dataset=dts,
        val_dataset=dts,
        pretrain=None,
        steps_per_epoch_override=2,
        use_tensorboard=False,
        mixed_precision=False,
    )

    model_zips = sorted(glob.glob(join(P.models_dir, "*", "*_epoch1.zip")))
    assert len(model_zips) == 1
    model_zip = model_zips[0]

    custom_objs = {
        "custom": _TinyCustomNet,
        "custom_classification": _TinyCustomLoss,
    }
    loaded_model = sf.model.load(model_zip, custom_objects=custom_objs)
    assert isinstance(loaded_model, torch.nn.Module)

    eval_res = P.evaluate(
        model_zip,
        outcomes="category",
        dataset=dts,
        batch_size=2,
        mixed_precision=False,
        custom_objects=custom_objs,
    )
    assert isinstance(eval_res, dict)


def test_patch_311_heatmap_extent_40x_no_drift():
    """Regression test for upstream issue #440 (progressive drift with tile_um='40x')."""
    from types import SimpleNamespace
    from slideflow.heatmap import calculate_heatmap_extent

    # Simulate a 40x SVS slide where mpp=0.2520 causes wsi.tile_um = int(256*0.2520) = 64
    # while true full_extract_px=256 and full_stride=128.
    wsi = SimpleNamespace(
        mpp=0.2520,
        tile_px=256,
        tile_um=64,
        stride_div=2,
        full_extract_px=256,
        full_stride=128,
        dimensions=(25600, 12800),
    )
    thumb = SimpleNamespace(size=(2048, 1024))
    grid = np.zeros((99, 199, 2), dtype=np.float32)

    left, right, bottom, top = calculate_heatmap_extent(wsi, thumb, grid)
    ratio_x = 25600 / 2048  # 12.5
    ratio_y = 12800 / 1024  # 12.5
    expected_offset = 256 / 2 - 128 / 2  # 64.0
    assert np.isclose(left, expected_offset / ratio_x)
    assert np.isclose(right, (expected_offset + 199 * 128) / ratio_x)
    assert np.isclose(bottom, (expected_offset + 99 * 128) / ratio_y)
    assert np.isclose(top, expected_offset / ratio_y)
