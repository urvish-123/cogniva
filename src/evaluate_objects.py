"""
Evaluate the full pipeline on the held-out object test set.

python -m src.evaluate_objects

Each folder in dataset/Objects/test/ is one object class. An image passes when
the pipeline returns type "object" with that class name.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from .config import ALLOWED_EXTENSIONS, OBJECT_TEST_DIR
from .database import get_all_objects
from .metadata import name_key
from .pipeline import recognize


def main(dataset: Path):
    names = {name_key(obj["name"]): obj["name"] for obj in get_all_objects()}
    per_class = defaultdict(lambda: [0, 0])
    times = []

    for object_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        expected = names.get(name_key(object_dir.name), object_dir.name)
        for image_path in sorted(object_dir.iterdir()):
            if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            result = recognize(image_path)
            times.append(result["processing_time"]["total_ms"])
            obj = result["object"] or {}
            predicted = obj.get("name") if result["type"] == "object" else result["type"]
            ok = predicted == expected

            per_class[expected][1] += 1
            per_class[expected][0] += ok
            print(
                f"{image_path.relative_to(dataset).as_posix()} | expected={expected} | "
                f"predicted={predicted} | score={obj.get('score')} | "
                f"{result['processing_time']['total_ms']} ms | {'PASS' if ok else 'FAIL'}"
            )

    print()
    for name, (correct, total) in per_class.items():
        print(f"{name}: {correct}/{total}")

    correct = sum(c for c, _ in per_class.values())
    total = sum(t for _, t in per_class.values())
    print(f"Accuracy: {correct / total * 100 if total else 0:.2f}% ({correct}/{total})")
    if times:
        print(f"Time per image: avg {sum(times) / len(times):.0f} ms, "
              f"max {max(times):.0f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(OBJECT_TEST_DIR))
    args = parser.parse_args()
    main(Path(args.dataset))
