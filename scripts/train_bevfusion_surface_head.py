"""Train the four-class surface head from frozen BEVFusion decoder caches.

English: run through scripts/run_python.sh. The final 1x1 layer is random;
only the two preceding Conv/BN/ReLU blocks inherit pretrained map weights.
Chinese / 中文：仅训练四类地图头；最后 1x1 分类层随机初始化，前两组卷积继承预训练。
No backbone is instantiated and no validation sample enters optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from bevfusion_native_camera import BEVSegmentationHead, load_checkpoint_subset

CLASSES = ("sidewalk", "road", "vegetation", "crosswalk")
GRID = {"frame": "reference LiDAR", "axis_order": ["class", "increasing_x", "increasing_y"],
        "shape": [4, 200, 200], "bounds_m": {"x": [-50, 50], "y": [-50, 50]},
        "resolution_m": 0.5, "decoder_shape": [512, 128, 128],
        "decoder_bounds_m": [-51.2, 51.2], "transform": "native BEVGridTransform, align_corners=False"}
THRESHOLD = 0.5


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def require_finite(value, description):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"Non-finite {description}")


def masked_focal_loss(logits, target, valid, alpha=0.25, gamma=2.0):
    """Independent sigmoid focal loss / (valid cells * four classes).

    Unknown cells contribute neither positives nor negatives. An all-unknown
    batch returns a differentiable exact zero, with zero gradients everywhere.
    """
    if logits.shape != target.shape or valid.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Incompatible logits, target, or valid shapes")
    if logits.shape[1] != 4:
        raise ValueError("Expected four independent surface classes")
    scores = logits.float()
    mask = valid.bool().unsqueeze(1)
    # Mask inputs before loss too: ignored cells cannot inject NaN into reduction.
    scores = torch.where(mask, scores, torch.zeros_like(scores))
    target = torch.where(mask, target.float(), torch.zeros_like(scores))
    bce = F.binary_cross_entropy_with_logits(scores, target, reduction="none")
    probability = scores.sigmoid()
    pt = probability * target + (1 - probability) * (1 - target)
    weight = alpha * target + (1 - alpha) * (1 - target)
    loss = weight * (1 - pt).pow(gamma) * bce
    denominator = valid.sum().to(scores.dtype).clamp_min(1) * logits.shape[1]
    return (loss * mask).sum() / denominator


def make_head(checkpoint=None, trainable="full"):
    # Load the ORIGINAL six-class head strictly BEFORE replacing its final layer.
    head = BEVSegmentationHead()
    loaded = load_checkpoint_subset(head, checkpoint, "heads.map.") if checkpoint is not None else 0
    inherited = list(head.classifier.children())[:6]
    head.classifier = nn.Sequential(*inherited, nn.Conv2d(512, 4, 1))
    head.classes = CLASSES
    if trainable == "final":
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        for parameter in head.classifier[-1].parameters():
            parameter.requires_grad_(True)
    return head, loaded


def set_train_mode(head, trainable):
    head.train()
    if trainable == "final":
        # Freeze BN running statistics as well as inherited parameters.
        for block in head.classifier[:6]:
            block.eval()


def manifest_path(cache_root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (cache_root / path).resolve()


def prepare_inputs(args):
    from bevfusion_surface_data import load_split_rows

    cache_root = args.cache_root.resolve()
    manifest_file = cache_root / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format") != "bevfusion-decoder-cache-v1":
        raise ValueError("Unsupported cache format")
    checkpoint = args.checkpoint or manifest_path(cache_root, manifest["checkpoint"])
    export_root = args.export_root or manifest_path(cache_root, manifest["export_root"])
    split_audit = args.split_audit or manifest_path(cache_root, manifest["split_audit"])
    checkpoint, export_root, split_audit = (p.resolve() for p in (checkpoint, export_root, split_audit))
    print(f"[provenance] hashing frozen checkpoint {checkpoint}", flush=True)
    checkpoint_hash = digest_file(checkpoint)
    if checkpoint_hash != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint SHA256 does not match cache manifest")
    rows = load_split_rows(export_root, split_audit)
    for field, file in (("split_audit_sha256", split_audit),
                        ("export_dataset_sha256", export_root / "dataset.json"),
                        ("export_classes_sha256", export_root / "classes.json")):
        if field in manifest and digest_file(file) != manifest[field]:
            raise ValueError(f"Cache provenance changed: {field}")
    if manifest.get("status", "completed") != "completed":
        raise ValueError("Cache must be completed before training")
    indexed = {}
    scene_sets = {}
    for split in ("train", "val"):
        indexed[split] = {}
        scene_sets[split] = set()
        for row in rows[split]:
            token, scene = str(row["sample_token"]), str(row["scene"])
            if token in indexed[split]:
                raise ValueError(f"Duplicate data token: {token}")
            indexed[split][token] = row
            scene_sets[split].add(scene)
    if scene_sets["train"] & scene_sets["val"] or indexed["train"].keys() & indexed["val"].keys():
        raise ValueError("Data split leakage: overlapping scenes or sample tokens")
    groups = {"train": [], "val": []}
    seen = set()
    for entry in manifest["entries"]:
        token, scene, split = str(entry["sample_token"]), str(entry["scene"]), entry["split"]
        if split not in groups or token in seen:
            raise ValueError(f"Invalid split or duplicate cache token: {token}, {split}")
        seen.add(token)
        row = indexed[split].get(token)
        if row is None or str(row["scene"]) != scene:
            raise ValueError(f"Cache token/scene/split mismatches official audited data: {token}")
        for field, key in (("labels_sha256", "labels"), ("frame_sha256", "frame")):
            source_file = (export_root / row[key]).resolve()
            if not source_file.is_relative_to(export_root):
                raise ValueError(f"Source path escapes export root: {key}")
            if field in entry and digest_file(source_file) != entry[field]:
                raise ValueError(f"Cached supervision/geometry changed: {token}, {field}")
        relative = Path(entry["feature_path"])
        if relative.is_absolute():
            raise ValueError("feature_path must be relative to cache_root")
        feature = (cache_root / relative).resolve()
        if not feature.is_relative_to(cache_root) or not feature.is_file():
            raise ValueError(f"Missing or escaping feature file: {relative}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", token):
            raise ValueError("Sample tokens must be safe file stems")
        groups[split].append({**entry, "feature_file": str(feature), "row": row})
    full_cached = {split: sorted(e["sample_token"] for e in groups[split]) for split in groups}
    cache_limit = manifest.get("settings", {}).get("max_per_split")
    if cache_limit is not None and (type(cache_limit) is not int or cache_limit < 1):
        raise ValueError("Invalid cache max_per_split")
    for split in groups:
        expected_rows = rows[split][:cache_limit] if cache_limit is not None else rows[split]
        expected = {r["sample_token"] for r in expected_rows}
        if set(full_cached[split]) != expected:
            raise ValueError(f"Cache {split} token set differs from audited rows and declared cache limit")
        if "sample_counts" in manifest and manifest["sample_counts"].get(split) != len(groups[split]):
            raise ValueError(f"Cache {split} sample count differs from manifest")
    for split, maximum in (("train", args.max_train_samples), ("val", args.max_val_samples)):
        groups[split].sort(key=lambda entry: entry["sample_token"])
        if maximum is not None:
            groups[split] = groups[split][:maximum]
        if not groups[split]:
            raise ValueError(f"No cached {split} samples available")
    split_identity = {
        "split_audit_sha256": digest_file(split_audit),
        "official_rows_sha256": {s: identity(sorted((t, str(r["scene"])) for t, r in indexed[s].items())) for s in groups},
        "official_scene_names": {s: sorted(scene_sets[s]) for s in groups},
        "cached_sample_tokens": full_cached,
        "selected_sample_tokens": {s: [e["sample_token"] for e in groups[s]] for s in groups},
        "selected_scenes": {s: sorted({e["scene"] for e in groups[s]}) for s in groups},
    }
    split_identity["sha256"] = identity(split_identity)
    return groups, {"cache_root": str(cache_root), "manifest_sha256": digest_file(manifest_file),
                    "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                    "export_root": str(export_root), "split_audit": str(split_audit),
                    "cache_settings": manifest.get("settings", {}), "split_identity": split_identity}


class FeatureDataset(Dataset):
    def __init__(self, entries, export_root):
        self.entries = entries
        self.export_root = Path(export_root)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        from bevfusion_surface_data import load_native_target

        entry = self.entries[index]
        payload = torch.load(entry["feature_file"], map_location="cpu", weights_only=True)
        for key in ("sample_token", "scene", "split"):
            if payload.get(key) != entry[key]:
                raise ValueError(f"Feature metadata mismatch for {entry['sample_token']}: {key}")
        features = payload["features"]
        if not isinstance(features, torch.Tensor) or features.shape != (512, 128, 128) or features.dtype != torch.float16:
            raise ValueError("Cache must contain an FP16 Tensor[512,128,128]")
        require_finite(features, f"features {entry['sample_token']}")
        data = load_native_target(self.export_root, entry["row"])
        target = torch.as_tensor(data["target"], dtype=torch.float32)
        valid = torch.as_tensor(data["valid"], dtype=torch.bool)
        labels = torch.as_tensor(data["labels"], dtype=torch.uint8)
        if target.shape != (4, 200, 200) or valid.shape != (200, 200) or labels.shape != (200, 200):
            raise ValueError("Target helper returned unexpected native grid shape")
        require_finite(target, "native target")
        if torch.any(labels > 4):
            raise ValueError("Unexpected label id outside unknown=0, classes=1..4")
        # Defense in depth: unknown must NEVER become a negative example.
        valid = valid & (labels != 0)
        if not torch.all((target == 0) | (target == 1)):
            raise ValueError("Expected binary one-hot targets")
        expected = F.one_hot(labels.long(), 5)[..., 1:].permute(2, 0, 1).float()
        if not torch.equal(target[:, valid], expected[:, valid]):
            raise ValueError("Target channel order disagrees with label ids 1..4")
        return {"features": features.detach(), "target": target, "valid": valid,
                "labels": labels, "sample_token": entry["sample_token"], "scene": entry["scene"]}


class MetricAccumulator:
    def __init__(self):
        self.tp = torch.zeros(4, dtype=torch.int64)
        self.fp = self.tp.clone()
        self.fn = self.tp.clone()
        self.support = self.tp.clone()
        self.confusion = torch.zeros((4, 4), dtype=torch.int64)
        self.valid_cells = 0
        self.loss_numerator = 0.0

    def update(self, logits, target, valid, labels, loss):
        predicted = logits.detach().float().sigmoid() >= THRESHOLD
        actual = target.bool()
        mask = valid[:, None].bool()
        dims = (0, 2, 3)
        self.tp += (predicted & actual & mask).sum(dims).cpu()
        self.fp += (predicted & ~actual & mask).sum(dims).cpu()
        self.fn += (~predicted & actual & mask).sum(dims).cpu()
        self.support += (actual & mask).sum(dims).cpu()
        gt = labels[valid].long() - 1
        pred = logits.detach().argmax(1)[valid].long()
        self.confusion += torch.bincount(gt * 4 + pred, minlength=16).reshape(4, 4).cpu()
        count = int(valid.sum().item())
        self.valid_cells += count
        self.loss_numerator += float(loss.detach().item()) * count

    def result(self):
        per_class = {}
        ious = []
        for i, name in enumerate(CLASSES):
            tp, fp, fn, support = (int(values[i]) for values in (self.tp, self.fp, self.fn, self.support))
            # Missing GT class is explicitly null; FP counts remain visible.
            iou = tp / (tp + fp + fn) if support else None
            if iou is not None:
                ious.append(iou)
            per_class[name] = {"iou": iou, "precision": tp / (tp + fp) if tp + fp else None,
                               "recall": tp / support if support else None,
                               "tp": tp, "fp": fp, "fn": fn, "support": support,
                               "predicted_positive": tp + fp, "union": tp + fp + fn}
        return {"loss": self.loss_numerator / self.valid_cells if self.valid_cells else None,
                "valid_cells": self.valid_cells, "threshold": THRESHOLD,
                "macro_iou": sum(ious) / len(ious) if ious else None,
                "macro_iou_present_classes": len(ious), "per_class": per_class,
                "independent_protocol": "Four independent sigmoid masks >=0.5; valid nonzero GT cells only; GT-absent IoU=null",
                "argmax_protocol": "Separate mutually exclusive argmax protocol, no probability threshold; rows=GT, columns=prediction",
                "argmax_confusion": self.confusion.tolist(),
                "argmax_accuracy": int(self.confusion.diag().sum()) / self.valid_cells if self.valid_cells else None,
                "class_order": list(CLASSES)}


def amp_context(args):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if args.amp else nullcontext()


def move_batch(batch, device):
    return (batch["features"].to(device=device, dtype=torch.float32, non_blocking=True),
            batch["target"].to(device, non_blocking=True),
            batch["valid"].to(device, non_blocking=True),
            batch["labels"].to(device, non_blocking=True))


def train_epoch(head, loader, optimizer, scaler, args, epoch):
    set_train_mode(head, args.trainable)
    metrics = MetricAccumulator()
    steps = args.overfit_steps or len(loader)
    if args.overfit_steps:
        fixed_batch = next(iter(loader))
        batches = (fixed_batch for _ in range(steps))
    else:
        batches = iter(loader)
    updates = 0
    for step, batch in enumerate(batches, 1):
        features, target, valid, labels = move_batch(batch, args.device)
        if not valid.any().item():
            print(f"[train] epoch={epoch} step={step}/{steps} skipped all-unknown batch", flush=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        with amp_context(args):
            logits = head(features, return_logits=True)
        require_finite(logits, "training logits")
        loss = masked_focal_loss(logits, target, valid, args.alpha)
        require_finite(loss, "training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in head.named_parameters():
            if parameter.grad is not None:
                require_finite(parameter.grad, f"gradient {name}")
        scaler.step(optimizer)
        scaler.update()
        updates += 1
        metrics.update(logits, target, valid, labels, loss)
        if step == 1 or step % args.log_every == 0 or step == steps:
            print(f"[train] epoch={epoch} step={step}/{steps} loss={float(loss.detach()):.6f} valid={int(valid.sum())}", flush=True)
    if updates == 0:
        raise ValueError("Training produced no updates: every selected sample was unknown")
    return {**metrics.result(), "optimizer_updates": updates}


@torch.inference_mode()
def evaluate(head, loader, args, epoch):
    head.eval()
    metrics = MetricAccumulator()
    for step, batch in enumerate(loader, 1):
        features, target, valid, labels = move_batch(batch, args.device)
        with amp_context(args):
            logits = head(features, return_logits=True)
        require_finite(logits, "validation logits")
        loss = masked_focal_loss(logits, target, valid, args.alpha)
        require_finite(loss, "validation loss")
        metrics.update(logits, target, valid, labels, loss)
        if step == 1 or step % args.log_every == 0 or step == len(loader):
            print(f"[val] epoch={epoch} step={step}/{len(loader)}", flush=True)
    result = metrics.result()
    if result["macro_iou"] is None:
        raise ValueError("Validation has no known surface cells; cannot select best head")
    return result


def cpu_state(head):
    return {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}


def save_checkpoint(path, head, optimizer, scaler, epoch, best_epoch, best_score, config, last):
    payload = {"format": "bevfusion-surface-head-v1", "head_state_dict": cpu_state(head),
               "classes": CLASSES, "model_grid": GRID,
               "frozen_checkpoint": config["provenance"]["checkpoint"],
               "frozen_checkpoint_sha256": config["provenance"]["checkpoint_sha256"],
               "train_split_identity": config["provenance"]["split_identity"],
               "initialization": config["initialization"], "trainable": config["trainable"],
               "epoch": epoch, "epochs": epoch, "planned_epochs": config["effective_epochs"],
               "best_epoch": best_epoch, "best_val_macro_iou": best_score,
               "threshold": THRESHOLD, "run_config": config}
    if last:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["grad_scaler_state_dict"] = scaler.state_dict()
    # The run directory is created exclusively; only this invocation writes here.
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def comparison_png(path, target, probability, valid):
    from PIL import Image, ImageDraw

    colors = np.array([[215, 85, 180], [65, 165, 95], [80, 130, 40], [245, 220, 75]], dtype=np.uint8)
    size, header = 360, 44
    canvas = Image.new("RGB", (3 * size, 4 * (size + header)), (18, 23, 30))
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(CLASSES):
        truth, pred = target[c].astype(bool), probability[c] >= THRESHOLD
        gt = np.full((200, 200, 3), 25, dtype=np.uint8)
        pp = gt.copy()
        error = np.full_like(gt, [35, 105, 65])
        gt[truth] = colors[c]
        pp[pred] = colors[c]
        error[pred & ~truth] = [235, 100, 50]
        error[~pred & truth] = [90, 125, 245]
        for col, (image, title) in enumerate(zip((gt, pp, error), ("GT", "mask prediction >=0.5", "error: FP orange / FN blue"))):
            image[~valid] = [125, 125, 125]
            # Native rows increase +X and columns +Y; flip BOTH axes for +X up, +Y left.
            panel = Image.fromarray(np.flip(image, axis=(0, 1)).copy()).resize((size, size), Image.Resampling.NEAREST)
            y = c * (size + header)
            canvas.paste(panel, (col * size, y + header))
            draw.text((col * size + 8, y + 6), f"{name}: {title}", fill="white")
            draw.text((col * size + 8, y + 23), "+X up, +Y left | unknown grey", fill="#bfc9d4")
    canvas.save(path)


@torch.inference_mode()
def save_examples(head, datasets, args):
    head.eval()
    records = []
    for split, dataset in datasets.items():
        # Spread examples through the deterministic sample ordering.
        indices = np.linspace(0, len(dataset) - 1, min(args.examples_per_split, len(dataset)), dtype=int)
        for index in indices:
            data = dataset[int(index)]
            token = data["sample_token"]
            destination = args.output_dir / "examples" / split / token
            destination.mkdir(parents=True, exist_ok=False)
            features = data["features"][None].to(args.device, dtype=torch.float32)
            with amp_context(args):
                logits = head(features, return_logits=True)
            require_finite(logits, "example logits")
            probability = logits.float().sigmoid()[0].cpu().numpy()
            target, valid, labels = (data[key].numpy() for key in ("target", "valid", "labels"))
            np.savez_compressed(destination / "prediction.npz", probabilities=probability,
                                masks=probability >= THRESHOLD, argmax_labels=probability.argmax(0).astype(np.uint8) + 1,
                                target=target, valid=valid, labels=labels, classes=np.asarray(CLASSES),
                                sample_token=np.asarray(token), scene=np.asarray(data["scene"]), split=np.asarray(split),
                                threshold=np.asarray(THRESHOLD), grid_json=np.asarray(json.dumps(GRID)))
            comparison_png(destination / "comparison.png", target, probability, valid)
            records.append({"split": split, "sample_token": token, "scene": data["scene"],
                            "prediction": str((destination / "prediction.npz").relative_to(args.output_dir)),
                            "image": str((destination / "comparison.png").relative_to(args.output_dir))})
            print(f"[example] {split} {token}", flush=True)
    return records


def curve_svg(history):
    width, height, pad = 760, 260, 35
    lines = []
    for split, color in (("train", "#79caff"), ("val", "#ffbf69")):
        values = [(r["epoch"], r[split]["macro_iou"]) for r in history if r[split]["macro_iou"] is not None]
        points = " ".join(f"{pad + (e-1) / max(1,len(history)-1)*(width-2*pad):.2f},{height-pad-v*(height-2*pad):.2f}" for e, v in values)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for e, v in values:
            x = pad + (e - 1) / max(1, len(history) - 1) * (width - 2 * pad)
            y = height - pad - v * (height - 2 * pad)
            lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Train and validation macro IoU"><path d="M35 25V225H735" fill="none" stroke="#aaa"/><text x="2" y="30" fill="white">1</text><text x="2" y="225" fill="white">0</text>{"".join(lines)}<text x="80" y="255" fill="#79caff">train (online)</text><text x="260" y="255" fill="#ffbf69">validation</text><text x="540" y="255" fill="white">epoch 1 to {len(history)}</text></svg>'


def write_report(output, config, history, metrics, examples):
    galleries = "".join(f'<figure><figcaption>{html.escape(r["split"])} / {html.escape(r["scene"])} / {html.escape(r["sample_token"])}</figcaption><a href="{r["prediction"]}">prediction.npz</a><img loading="lazy" src="{r["image"]}" alt="Four classes: GT, threshold prediction and error"/></figure>' for r in examples)
    details = html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))
    initialization = html.escape(config["initialization"])
    document = f'''<!doctype html><html lang="en" id="top"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BEVFusion four-class surface head</title>
<style>body{{background:#111820;color:#e6edf3;font:16px/1.65 sans-serif;max-width:1200px;margin:30px auto;padding:0 20px}}a{{color:#8acbff}}img,svg{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#1a2530;padding:18px}}figure{{margin:30px 0}}table{{border-collapse:collapse}}td,th{{border:1px solid #566;padding:8px}}</style>
<h1>BEVFusion four-class surface head</h1><a href="#zh">中文说明</a>
<p>Cached frozen camera + LiDAR + fuser + decoder features. Trainable mode: {config['trainable']}. {initialization}</p>
<p>Official scene-level train/validation membership is checked against the split audit. Only selected training tokens enter optimization. Debug limits or overfit steps produce a limited smoke run; cached sample coverage is recorded in run_config.json. The pretrained checkpoint's historical training data cannot be established from its weights.</p>
<p>Best epoch: {metrics['best_epoch']}; validation macro IoU: {metrics['best_val_macro_iou']:.6f}. Figures use best.pth. Green error cells mean agreement only on known GT; grey unknown cells are excluded.</p>
<p>Four independent sigmoid masks at fixed threshold 0.5 give class IoU, precision and recall. GT-absent class IoU is null and excluded from macro IoU; false positives remain in counts. Argmax confusion uses a separate mutually exclusive four-class protocol, valid cells only. Threshold masks may overlap or leave cells unassigned.</p>
<p>Train curves are measured during updates; validation is evaluated after each epoch. Labels are scene-fused teacher supervision and do not establish independently annotated ground truth. No original six-class taxonomy baseline is claimed.</p>
{curve_svg(history)}<p><a href="run_config.json">Run configuration</a> · <a href="history.jsonl">History</a> · <a href="metrics.json">Metrics</a> · <a href="best.pth">Best head</a> · <a href="last.pth">Last head + optimizer</a></p>
<h2>Predictions: GT / independent mask / error</h2>{galleries}
<h2>Metrics</h2><pre>{details}</pre>
<section id="zh" lang="zh-CN"><h2>中文说明</h2><a href="#top">English / 返回顶部</a>
<p>冻结预训练相机、激光雷达、融合器与解码器；只读取缓存并训练地图头。两组 3×3 卷积与 BN 继承原六类预训练头，最后 1×1 四类层随机初始化，不能称为完整预训练四类头。默认训练整个新头；final 模式冻结前面参数和 BN 统计。</p>
<p>固定官方按场景 train/val 划分，验证集不参与梯度或优化；缓存覆盖、调试截断及实际 token 清单记录于配置。仅凭预训练权重不能证明其历史训练样本清单。类别顺序：人行道、道路、植被、人行横道。</p>
<p>四个独立 sigmoid 掩码统一使用阈值 0.5；IoU、精确率与召回率只在有效且标签非零的区域计算。GT 缺失类 IoU 为 null、从宏平均排除，误报计数仍保留。argmax 混淆矩阵采用另一套四类互斥协议，不能混同阈值结果。未知区为灰色，不自动算正确。</p>
<p>曲线训练指标为优化过程在线统计，验证指标为每轮结束统计；示例使用最佳验证宏 IoU 权重。橙色为误报、蓝色为漏报、绿色为有效区一致。场景融合标签是教师监督，并非独立人工真值；调试或过拟合运行不能证明泛化效果。</p></section></html>'''
    (output / "index.html").write_text(document, encoding="utf-8")


def self_test(device):
    torch.manual_seed(42)
    logits = torch.randn(1, 4, 8, 8, device=device, requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.zeros((1, 8, 8), device=device, dtype=torch.bool)
    loss = masked_focal_loss(logits, target, valid)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(logits.grad).item() == 0
    logits.grad = None
    valid[:, :4, :4] = True
    target[:, 1, :4, :4] = 1
    loss = masked_focal_loss(logits, target, valid)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :, 4:, :]).item() == 0
    assert torch.count_nonzero(logits.grad[:, :, :, 4:]).item() == 0
    assert torch.count_nonzero(logits.grad[:, :, :4, :4]).item() > 0
    print("[self-test] all-unknown loss=0, all gradients=0; mixed unknown gradients=0", flush=True)
    original = BEVSegmentationHead()
    checkpoint = {"state_dict": {"heads.map." + k: v for k, v in original.state_dict().items()}}
    head, loaded = make_head(checkpoint)
    assert torch.equal(head.classifier[0].weight, original.classifier[0].weight)
    assert head.classifier[-1].out_channels == 4 and tuple(head.classes) == CLASSES
    del original, checkpoint
    head = head.to(device)
    features = torch.randn(1, 512, 128, 128, device=device, dtype=torch.float32)
    target = torch.zeros((1, 4, 200, 200), device=device)
    target[:, 1] = 1
    valid = torch.ones((1, 200, 200), device=device, dtype=torch.bool)
    valid[:, :10] = False
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    previous = head.classifier[-1].weight.detach().clone()
    logits = head(features, return_logits=True)
    assert logits.shape == (1, 4, 200, 200)
    loss = masked_focal_loss(logits, target, valid)
    require_finite(loss, "self-test loss")
    loss.backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        require_finite(parameter.grad, name)
    assert head.classifier[0].weight.grad.abs().sum().item() > 0
    optimizer.step()
    assert not torch.equal(previous, head.classifier[-1].weight)
    print(f"[self-test] strict six-class load={loaded} tensors; full native 512x128x128 -> 4x200x200 backward/update passed, loss={loss.item():.6f}", flush=True)
    head.zero_grad(set_to_none=True)
    final_head, _ = make_head(trainable="final")
    set_train_mode(final_head, "final")
    assert not final_head.classifier[1].training and not final_head.classifier[4].training
    assert sum(p.numel() for p in final_head.parameters() if p.requires_grad) == 2052
    meter = MetricAccumulator()
    small_logits = torch.zeros(1, 4, 2, 2)
    small_target = torch.zeros_like(small_logits)
    small_target[:, 1] = 1
    small_valid = torch.ones((1, 2, 2), dtype=torch.bool)
    small_labels = torch.full((1, 2, 2), 2, dtype=torch.uint8)
    meter.update(small_logits, small_target, small_valid, small_labels, torch.tensor(0.0))
    result = meter.result()
    assert result["per_class"]["sidewalk"]["iou"] is None
    assert result["per_class"]["sidewalk"]["fp"] == 4
    assert result["macro_iou"] == 1.0 and sum(map(sum, result["argmax_confusion"])) == 4
    print("[self-test] final-only BN freeze and missing-class metric protocols passed", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--overfit-steps", type=int, default=0,
                        help="Smoke test: repeat the first training batch N times for ONE epoch; val stays held out")
    parser.add_argument("--trainable", choices=("full", "final"), default="full")
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal positive alpha; negatives use 1-alpha; gamma fixed at 2")
    parser.add_argument("--amp", action="store_true", help="Optional CUDA FP16 autocast; default is FP32 head training")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--export-root", type=Path)
    parser.add_argument("--split-audit", type=Path)
    parser.add_argument("--examples-per-split", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic loss/gradient/full-head tests without a cache")
    args = parser.parse_args()
    if not args.self_test and (args.cache_root is None or args.output_dir is None):
        parser.error("--cache-root and --output-dir are required for training")
    if min(args.epochs, args.batch_size, args.log_every, args.examples_per_split) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, log-every, examples-per-split and lr must be positive")
    if args.overfit_steps < 0 or args.num_workers < 0 or not 0 < args.alpha < 1:
        parser.error("Invalid overfit-steps, num-workers or focal alpha")
    for value in (args.max_train_samples, args.max_val_samples):
        if value is not None and value < 1:
            parser.error("Debug sample limits must be positive")
    if args.amp and torch.device(args.device).type != "cuda":
        parser.error("--amp requires a CUDA device")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run with scripts/run_python.sh")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    print(f"[runtime] torch={torch.__version__} device={args.device} amp={args.amp}", flush=True)
    if args.self_test:
        self_test(args.device)
        return
    args.output_dir = args.output_dir.resolve()
    # Refuse any existing run, including empty directories.
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {args.output_dir}")
    groups, provenance = prepare_inputs(args)
    head, loaded = make_head(provenance["checkpoint"], args.trainable)
    head = head.to(args.device)
    datasets = {s: FeatureDataset(entries, provenance["export_root"]) for s, entries in groups.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(dataset, batch_size=args.batch_size, shuffle=(s == "train"),
                             num_workers=args.num_workers, pin_memory=torch.device(args.device).type == "cuda",
                             generator=generator if s == "train" else None)
               for s, dataset in datasets.items()}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"format": "bevfusion-surface-training-v1", "classes": list(CLASSES), "model_grid": GRID,
                   "provenance": provenance, "effective_epochs": 1 if args.overfit_steps else args.epochs,
                   "initialization": "Strict original six-class heads.map.* checkpoint load; first six classifier modules inherited; final Conv2d(512,4,1) randomly initialized",
                   "strictly_loaded_original_head_tensors": loaded,
                   "trainable_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad),
                   "frozen_backbone": "Not instantiated: FP16 cached camera+LiDAR+fuser+decoder features, converted to FP32 for head",
                   "loss": {"name": "masked sigmoid focal", "gamma": 2, "alpha": args.alpha,
                            "reduction": "sum / (known valid cells * 4)", "unknown": "label 0 excluded from positives and negatives"},
                   "threshold": THRESHOLD, "torch_version": str(torch.__version__),
                   "num_threads": torch.get_num_threads(),
                   "tf32": {"cudnn": torch.backends.cudnn.allow_tf32, "matmul": torch.backends.cuda.matmul.allow_tf32},
                   "device_name": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else str(args.device),
                   "sample_counts": {s: len(ds) for s, ds in datasets.items()},
                   "debug_run": bool(args.overfit_steps or args.max_train_samples or args.max_val_samples or provenance["cache_settings"].get("max_per_split"))})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump_json(args.output_dir / "run_config.json", config)
    optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    history = []
    best_score, best_epoch = -math.inf, 0
    started = time.monotonic()
    print(f"[start] train={len(datasets['train'])} val={len(datasets['val'])} parameters={config['trainable_parameters']} output={args.output_dir}", flush=True)
    for epoch in range(1, config["effective_epochs"] + 1):
        train = train_epoch(head, loaders["train"], optimizer, scaler, args, epoch)
        val = evaluate(head, loaders["val"], args, epoch)
        row = {"epoch": epoch, "elapsed_seconds": time.monotonic() - started, "train": train, "val": val}
        history.append(row)
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if val["macro_iou"] > best_score:
            best_score, best_epoch = val["macro_iou"], epoch
            save_checkpoint(args.output_dir / "best.pth", head, optimizer, scaler, epoch, best_epoch, best_score, config, False)
        save_checkpoint(args.output_dir / "last.pth", head, optimizer, scaler, epoch, best_epoch, best_score, config, True)
        metrics = {"classes": list(CLASSES), "best_epoch": best_epoch, "best_val_macro_iou": best_score,
                   "best_val": history[best_epoch - 1]["val"], "last_epoch": epoch,
                   "last_train": train, "last_val": val, "completed_epochs": len(history),
                   "selection": "Official-val macro IoU of independent >=0.5 masks; ignore GT-absent classes",
                   "debug_run": config["debug_run"], "model_grid": GRID}
        dump_json(args.output_dir / "metrics.json", metrics)
        print(f"[epoch] {epoch} train_loss={train['loss']:.6f} val_loss={val['loss']:.6f} val_macroIoU={val['macro_iou']:.6f} best_epoch={best_epoch}", flush=True)
    best = torch.load(args.output_dir / "best.pth", map_location="cpu", weights_only=True)
    head.load_state_dict(best["head_state_dict"], strict=True)
    examples = save_examples(head, datasets, args)
    metrics["examples"] = examples
    metrics["examples_checkpoint"] = "best.pth"
    metrics["elapsed_seconds"] = time.monotonic() - started
    dump_json(args.output_dir / "metrics.json", metrics)
    write_report(args.output_dir, config, history, metrics, examples)
    print(f"[done] best_epoch={best_epoch} val_macroIoU={best_score:.6f} report={args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
