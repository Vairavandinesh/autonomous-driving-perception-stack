from pathlib import Path
from collections import Counter
import csv

# KITTI label files contain one object per line.
label_dir = Path("datasets/KITTI/label_2")
output_file = Path("reports/kitti_class_counts.csv")

if not label_dir.exists():
    print("KITTI labels not found yet.")
    print("Expected folder: datasets/KITTI/label_2")
    print("Download/extract the KITTI training labels before running analysis.")
else:
    counts = Counter()
    total = 0

    for label_file in label_dir.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            fields = line.split()
            if fields:
                counts[fields[0]] += 1
                total += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "object_count"])
        writer.writerows(counts.most_common())

    print(f"Total labeled objects: {total}")
    for class_name, count in counts.most_common():
        print(f"{class_name}: {count}")
    print(f"Saved report: {output_file}")