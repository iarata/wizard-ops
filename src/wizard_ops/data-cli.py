import os
import re
from pathlib import Path
from typing import Annotated

import kagglehub as kh
import pandas as pd
import typer

app = typer.Typer(help="Commands to manage the Nutrition dataset")

# Path to metadata files
METADATA_DIR = "src/wizard_ops/metadata/"
DISH_METADATA_CAFE1 = "src/wizard_ops/metadata/dish_metadata_cafe1.csv"
DISH_METADATA_CAFE2 = "src/wizard_ops/metadata/dish_metadata_cafe2.csv"
INGREDIENTS_METADATA = "src/wizard_ops/metadata/ingredients_metadata.csv"

_DISH_TOTAL_COLUMNS = [
    "dish_id",
    "total_calories",
    "total_mass",
    "total_fat",
    "total_carb",
    "total_protein",
    "num_ingrs",
]


def load_dish_metadata(
    cafe1_path: str | Path = DISH_METADATA_CAFE1,
    cafe2_path: str | Path = DISH_METADATA_CAFE2,
) -> pd.DataFrame:
    """Load dish-level nutrition metadata.

    The raw CSVs are *ragged*: after the first fixed set of dish-level fields,
    each dish contains a variable-length list of per-ingredient fields.

    We only need the dish-level totals plus `num_ingrs`.
    `num_ingrs` is computed from the remaining fields.
    """

    def _read_totals(csv_path: str | Path) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # First 6 fields are fixed; remainder is per-ingredient blocks.
                parts = line.split(",", 6)
                if len(parts) < 6:
                    continue

                dish_id = parts[0]
                try:
                    totals = [float(x) for x in parts[1:6]]
                except ValueError:
                    continue

                remainder = parts[6] if len(parts) == 7 else ""
                remainder_fields = remainder.split(",") if remainder else []

                # Each ingredient contributes 7 fields: id, name, grams, calories, fat, carb, protein
                num_ingrs = len(remainder_fields) // 7
                rows.append(
                    {
                        "dish_id": dish_id,
                        "total_calories": totals[0],
                        "total_mass": totals[1],
                        "total_fat": totals[2],
                        "total_carb": totals[3],
                        "total_protein": totals[4],
                        "num_ingrs": int(num_ingrs),
                    }
                )

        return pd.DataFrame(rows, columns=_DISH_TOTAL_COLUMNS)

    return pd.concat([
        _read_totals(cafe1_path),
        _read_totals(cafe2_path),
    ], ignore_index=True)


def load_ingredients_metadata(path: str | Path = INGREDIENTS_METADATA) -> pd.DataFrame:
    """Load ingredient-level metadata."""
    return pd.read_csv(path)




# MARK: - Typer CLI commands to download the dataset from Kaggle
@app.command("download")
def download(dir: Annotated[str, typer.Option("--dir", "-d", help="Directory to download the dataset to")]):
    """Download the Nutrition Dataset from Kaggle."""
    path = Path(dir)
    try:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            kh_path = kh.dataset_download(
                "zygmuntyt/nutrition5k-dataset-side-angle-images",                
            )
            os.rename(kh_path, path)
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
    return path
    typer.echo(f"Dataset downloaded to {dir}")
    
# MARK: - CLI to generate unified dish metadata CSV/parquet
@app.command("generate-metadata")
def generate_metadata(
    data_dir: Annotated[str, typer.Option("--data-dir", "-d", help="Directory where the dataset is stored")],
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory for the unified metadata file")],
    cleanup: Annotated[bool, typer.Option("--cleanup", "-c", help="Whether to preprocess the metadata before saving")] = True,
    frame: Annotated[str, typer.Option("--frame", "-f", help="Frame type to save (e.g., 'A', 'B' ...)")] = "A",
    max_imgs_per_frame: Annotated[int, typer.Option("--max-imgs-per-frame", "-m", help="Maximum number of images per frame type to consider")] = 2,
):
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dishes_available_data = []
    for dish_dir in data_path.iterdir():
        if dish_dir.is_dir():
            dishes_available_data.append(dish_dir.name)
    print(f"Found {len(dishes_available_data)} dishes with available data.")
    
    # Load dish-level totals and computed ingredient count.
    # (The raw dish metadata CSVs are ragged and cannot be parsed by pandas' default CSV reader.)
    dish_meta = load_dish_metadata()
    print(f"Loaded dish metadata with shape: {dish_meta.shape}")

    unified = dish_meta[dish_meta["dish_id"].isin(dishes_available_data)].reset_index(drop=True)
    print(f"Unified metadata shape after filtering: {unified.shape}")
    
    # When frame is set, go through all dishes' folder in data_path and check if the specified frame type exists.
    # Sample check data.nosync/dish_1576258657/frames_sampled30/camera_A_frame_001.jpeg -> frame type A
    # When -m is set the dish must have exactly that number of images for the specified frame type.
    if frame:
        valid_dish_ids = []
        for dish_id in unified["dish_id"]:
            dish_dir = data_path / dish_id / "frames_sampled30"
            if not dish_dir.exists():
                continue
            frame_images = [
                img_file for img_file in dish_dir.iterdir()
                if re.match(rf"camera_{frame}_.*\.jpeg$", img_file.name)
            ]
            if len(frame_images) == max_imgs_per_frame:
                valid_dish_ids.append(dish_id)
                
        
        unified = unified[unified["dish_id"].isin(valid_dish_ids)].reset_index(drop=True)
        print(f"Unified metadata shape after filtering for frame {frame}: {unified.shape}")

    output_file = output_path / f"dish_metadata_{frame}.csv"
    unified.to_csv(output_file, index=False)
    print(f"Saved unified dish metadata to: {output_file}")
    
    

if __name__ == "__main__":
    app()
