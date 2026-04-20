from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from lab_utils.visualization import plot_class_balance, plot_numeric_distribution
SEED = 1234
SPLITS = ('train', 'val', 'test')
LABELS = ('cat', 'dog')
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
from pathlib import Path
import os

def list_image_paths_for_group(data_root: Path, split: str, label: str) -> list[Path]:
    folder = Path(data_root) / split / label
    image_paths = []
    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(folder / file)
    return image_paths

def inspect_image_file(path: Path) -> tuple[int, int, float]:
    img = Image.open(path).convert('RGB')
    arr = np.array(img) / 255.0
    height, width, _ = arr.shape
    mean_intensity = arr.mean()
    return (width, height, mean_intensity)

def make_metadata_row(path: Path, data_root: Path, split: str, label: str) -> dict[str, object]:
    width, height, mean_intensity = inspect_image_file(path)
    relative_path = os.path.relpath(path, data_root)
    return {'filepath': relative_path, 'label': label, 'split': split, 'width': width, 'height': height, 'mean_intensity': mean_intensity}

def build_metadata_from_folders(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in LABELS:
            paths = list_image_paths_for_group(data_root, split, label)
            rows.extend((make_metadata_row(p, data_root, split, label) for p in paths))
    return pd.DataFrame(rows).sort_values(['split', 'label', 'filepath']).reset_index(drop=True)

def load_metadata_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = ['filepath', 'label', 'split', 'width', 'height', 'mean_intensity']
    return df

def summarize_metadata(frame: pd.DataFrame) -> dict[str, object]:
    return {'rows': len(frame), 'columns': frame.columns.tolist(), 'class_counts': frame['label'].value_counts(), 'split_counts': frame['split'].value_counts()}

def build_label_split_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(frame['label'], frame['split'])
    return table

def audit_metadata(frame: pd.DataFrame) -> dict[str, object]:
    labels = {'cat', 'dog'}
    missing_values = frame.isnull().sum().to_dict()
    duplicate_filepaths = frame['filepath'].duplicated().sum()
    bad_labels = frame.loc[~frame['label'].isin(labels), 'label'].unique().tolist()
    non_positive_sizes = frame[(frame['width'] <= 0) | (frame['height'] <= 0)].shape[0]
    return {'missing_values': missing_values, 'duplicate_filepaths': duplicate_filepaths, 'bad_labels': bad_labels, 'non_positive_sizes': non_positive_sizes}

def add_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['pixel_count'] = result['width'] * result['height']
    result['aspect_ratio'] = result['width'] / result['height']
    result['brightness_band'] = pd.qcut(result['mean_intensity'], q=4, labels=['darkest', 'dim', 'bright', 'brightest'])
    ref_size = 64 * 64

    def size_category(pixels):
        if pixels < ref_size:
            return 'small'
        elif pixels == ref_size:
            return 'medium'
        else:
            return 'large'
    result['size_bucket'] = result['pixel_count'].apply(size_category)
    return result

def build_split_characteristics_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby('split')[['width', 'height', 'pixel_count', 'mean_intensity']].mean().rename(columns={'width': 'avg_width', 'height': 'avg_height', 'pixel_count': 'avg_pixel_count', 'mean_intensity': 'avg_mean_intensity'})

def sample_balanced_by_split_and_label(frame: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby(['split', 'label']):
        pieces.append(group.sample(n=min(len(group), n_per_group), random_state=seed))
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sort_values(['split', 'label', 'filepath']).reset_index(drop=True)
sample_size_per_group = 5
