from dataclasses import dataclass
from pathlib import Path

from .config import CLASS_NAMES, IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def load_dataset(dataset_dir: Path) -> list[ImageRecord]:
    """
    Load only the Banana Class_A, Class_B and Defect images.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"\nBanana dataset not found:\n{dataset_dir}\n\n"
            "Expected:\n"
            "Fruits_Data/Banana/Class_A\n"
            "Fruits_Data/Banana/Class_B\n"
            "Fruits_Data/Banana/Defect"
        )

    records: list[ImageRecord] = []

    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise ValueError(
                f"No images found in: {class_dir}"
            )

        records.extend(
            ImageRecord(path=path, label=class_name)
            for path in images
        )

    return records


def print_dataset_summary(records: list[ImageRecord]) -> None:
    print("\n" + "=" * 55)
    print("BANANA DATASET SUMMARY")
    print("=" * 55)

    for class_name in CLASS_NAMES:
        count = sum(
            record.label == class_name
            for record in records
        )
        print(f"{class_name:<12}: {count:>4} images")

    print("-" * 55)
    print(f"{'Total':<12}: {len(records):>4} images")
