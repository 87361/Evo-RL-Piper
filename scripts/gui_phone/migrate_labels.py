import os
import csv
import json
from pathlib import Path
from config import EPISODE_SCAN_ROOTS
from data_ops import scan_lerobot_datasets, load_csv, write_csv, categories_path, load_categories, write_categories

def migrate_labels():
    replaced_count = 0
    for root in EPISODE_SCAN_ROOTS:
        if not os.path.exists(root):
            continue
        for ds in scan_lerobot_datasets(Path(root)):
            ds_path = Path(ds['dataset_root'])
            label_csv = ds_path / "task_labels.csv"
            if not label_csv.exists():
                continue
            
            labels = load_csv(label_csv)
            cat_path = categories_path(label_csv)
            categories = load_categories(cat_path, labels)
            
            changed = False
            for ep, row in labels.items():
                if row.get("label") == "A":
                    row["label"] = "shirt open middle and catch"
                    changed = True
                    replaced_count += 1
                    
            if changed:
                if "shirt open middle and catch" not in categories:
                    categories.append("shirt open middle and catch")
                if "A" in categories:
                    categories.remove("A")
                write_categories(cat_path, categories)
                write_csv(label_csv, labels)
                print(f"Migrated labels in {ds_path.name}")
                
    print(f"Total replaced: {replaced_count}")

if __name__ == "__main__":
    migrate_labels()
