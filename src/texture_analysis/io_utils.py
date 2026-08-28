from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .config import IMAGE_EXTENSIONS, MASK_KEYWORDS


@dataclass(frozen=True)
class PairRecord:
    fruit: str
    category: str
    image: str
    relative_path: str
    processed_path: str | None
    mask_path: str | None
    status: str


def find_colour_csv(project_root: Path, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = project_root / p
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Colour feature CSV not found: {p}")

    candidates = [
        project_root / "results" / "colour_features.csv",
        project_root / "results" / "colour_analysis" / "colour_features.csv",
        project_root / "results" / "features" / "colour_features.csv",
        project_root / "colour_features.csv",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    matches = list((project_root / "results").rglob("colour_features.csv")) if (project_root / "results").exists() else []
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        "Could not find colour_features.csv automatically. "
        "Pass --colour-csv <path-to-colour_features.csv>."
    )


def read_colour_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"fruit", "category", "image", "relative_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"colour_features.csv is missing columns: {sorted(missing)}")
    return df


def _is_mask_path(path: Path) -> bool:
    text = (path.stem + " " + " ".join(path.parts)).lower()
    return any(k in text for k in MASK_KEYWORDS)


def _all_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _score_processed(candidate: Path, row: pd.Series) -> int:
    score = 0
    target_name = str(row["image"])
    target_stem = Path(target_name).stem
    rel = str(row.get("relative_path", "")).replace("\\", "/")
    rel_stem = Path(rel).stem if rel else target_stem

    if candidate.name.lower() == target_name.lower():
        score += 100
    if candidate.stem.lower() == target_stem.lower():
        score += 80
    if _norm(candidate.stem) == _norm(target_stem):
        score += 50
    if rel_stem.lower() in candidate.stem.lower():
        score += 25
    if str(row["fruit"]).lower() in str(candidate).lower():
        score += 12
    if str(row["category"]).lower() in str(candidate).lower():
        score += 8
    return score


def _score_mask(candidate: Path, row: pd.Series) -> int:
    score = 0
    target_stem = Path(str(row["image"])).stem
    cand_norm = _norm(candidate.stem)
    target_norm = _norm(target_stem)

    if target_norm and target_norm in cand_norm:
        score += 100
    if cand_norm.startswith(target_norm) or cand_norm.endswith(target_norm):
        score += 30
    if "roimask" in cand_norm:
        score += 25
    elif "mask" in cand_norm:
        score += 20
    elif "roi" in cand_norm:
        score += 15
    if str(row["fruit"]).lower() in str(candidate).lower():
        score += 10
    if str(row["category"]).lower() in str(candidate).lower():
        score += 6
    return score


def pair_processed_and_masks(colour_df: pd.DataFrame, preprocessing_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not preprocessing_dir.exists():
        raise FileNotFoundError(f"Preprocessing folder not found: {preprocessing_dir}")

    files = _all_images(preprocessing_dir)
    processed_files = [p for p in files if not _is_mask_path(p)]
    mask_files = [p for p in files if _is_mask_path(p)]

    # Fast exact-name index first.
    by_name: dict[str, list[Path]] = {}
    for p in processed_files:
        by_name.setdefault(p.name.lower(), []).append(p)

    records: list[dict] = []
    pairing: list[PairRecord] = []

    for _, row in colour_df.iterrows():
        exact = by_name.get(str(row["image"]).lower(), [])
        if exact:
            proc_candidates = exact
        else:
            target_norm = _norm(Path(str(row["image"])).stem)
            proc_candidates = [p for p in processed_files if target_norm in _norm(p.stem)]

        processed = max(proc_candidates, key=lambda p: _score_processed(p, row), default=None)

        target_norm = _norm(Path(str(row["image"])).stem)
        local_masks = [p for p in mask_files if target_norm in _norm(p.stem)]
        if processed is not None:
            # Same/nearby directory gets priority through scoring by appending it twice.
            near = [p for p in local_masks if p.parent == processed.parent]
            mask_candidates = near + local_masks
        else:
            mask_candidates = local_masks
        mask = max(mask_candidates, key=lambda p: _score_mask(p, row), default=None)

        if processed is None and mask is None:
            status = "missing_image_and_mask"
        elif processed is None:
            status = "missing_processed_image"
        elif mask is None:
            status = "missing_roi_mask"
        else:
            status = "ok"
            rec = row.to_dict()
            rec["processed_path"] = str(processed.resolve())
            rec["mask_path"] = str(mask.resolve())
            records.append(rec)

        pairing.append(
            PairRecord(
                fruit=str(row["fruit"]),
                category=str(row["category"]),
                image=str(row["image"]),
                relative_path=str(row["relative_path"]),
                processed_path=str(processed.resolve()) if processed else None,
                mask_path=str(mask.resolve()) if mask else None,
                status=status,
            )
        )

    paired_df = pd.DataFrame(records)
    report_df = pd.DataFrame([r.__dict__ for r in pairing])
    return paired_df, report_df


def load_processed_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read processed image: {path}")
    return img


def load_binary_mask(path: str | Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read ROI mask: {path}")
    if target_shape is not None and mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def ensure_dirs(*paths: Iterable[Path] | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
