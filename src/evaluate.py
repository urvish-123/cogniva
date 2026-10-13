"""
Simple CSV-driven evaluation.

Input CSV columns:
image,expected_name,case

expected_name is a celebrity name, "Unknown", or a status such as
"multiple_faces" / "no_face".

Example:
celebrity_dataset/Virat_Kohli/001.jpg,Virat Kohli,Celebrity dataset image
test_images/unknown_marmik_01.jpg,Unknown,Unknown person
test_images/multiple_faces_test.jpg,multiple_faces,Multiple faces
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .config import TEST_CASES_CSV
from .recognize import recognize_image


def main(csv_path: Path):
    with csv_path.open(encoding="utf-8") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["image"] and not r["image"].startswith("#")
        ]

    total = 0
    correct = 0
    per_case = defaultdict(lambda: [0, 0])

    for row in rows:
        total += 1
        result = recognize_image(Path(row["image"]))
        predicted = result.get("name") or result["status"]
        expected = row["expected_name"]
        ok = predicted.lower() == expected.lower()
        case = row.get("case") or "General"

        per_case[case][1] += 1
        if ok:
            correct += 1
            per_case[case][0] += 1

        print(
            f"{row['image']} | expected={expected} | "
            f"predicted={predicted} | score={result.get('score')} | "
            f"{'PASS' if ok else 'FAIL'}"
        )

    print()
    for case, (case_correct, case_total) in per_case.items():
        print(f"{case}: {case_correct}/{case_total}")

    accuracy = (correct / total * 100) if total else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(TEST_CASES_CSV))
    args = parser.parse_args()
    main(Path(args.csv))
