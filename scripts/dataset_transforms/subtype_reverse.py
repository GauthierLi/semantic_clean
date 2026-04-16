"""Convert corrected subtype manual-review JSON back into `.npy` samples.

This reverse converter uses the original `.npy` input as a template because the manual-review
JSON produced by `subtype.py --manual` does not retain enough information to reconstruct every
field from scratch (for example bbox and some ambiguous subtype fields).

The script keeps non-label fields from the original samples and rewrites subtype-related label
columns so that the corrected JSON categories can be converted back into `.npy` form for
round-trip validation.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np


MAIN_CATEGORY_TO_GT_CLASS = {
    'BACKGROUND': 0,
    'CAR': 1,
    'TRUCK': 2,
    'VAN': 3,
    'BUS': 4,
    'CONSTRUCTION_VEHICLE': 13,
    'CAT': 14,
    'TRAFFIC_CONE': 9,
    'BARRIER_DELINERATOR': 15,
}

MAIN_CATEGORY_PRIORITY = [
    'BACKGROUND',
    'CAR',
    'TRUCK',
    'VAN',
    'BUS',
    'CONSTRUCTION_VEHICLE',
    'CAT',
    'TRAFFIC_CONE',
    'BARRIER_DELINERATOR',
]

SV_CATEGORY_TO_VALUE = {
    'SV_NONE': 0,
    'SV_POLICE_CAR': 1,
    'SV_FIRE_ENGINE': 3,
    'SV_AMBULANCE': 4,
    'SV_SCHOOL_BUS': 6,
    'SV_SPRINKLER': 10,
    'SV_SPRINKLER_WO_BRUSH': 13,
    'SV_SANITIZATION_VEHICLE': 16,
}

SLOPE_CATEGORY_TO_VALUE = {
    'WITH_SLOPE': 14,
    'WO_SLOPE': 15,
}

DOOR_CATEGORY_TO_POSITION = {
    'DOOR_STATUS_BACK': 'back',
    'DOOR_STATUS_LEFT': 'left',
    'DOOR_STATUS_RIGHT': 'right',
    'DOOR_STATUS_FRONT': 'front',
}

SEGMENT_CATEGORIES = {
    'SEGMENT_HEAD': 'head',
    'SEGMENT_TAIL': 'tail',
    'SEGMENT_LEFT': 'left',
    'SEGMENT_RIGHT': 'right',
}

ROBOTAXI_CATEGORY_TO_VALUE = {
    'ROBOTAXI_NO': 0,
    'ROBOTAXI_RT5': 1,
    'ROBOTAXI_RT6': 2,
}

STOP_SIGN_CATEGORY_TO_VALUE = {
    'STOP_SIGN_UNSEEN': 0,
    'STOP_SIGN_ON': 1,
    'STOP_SIGN_OFF': 2,
}

PROFESSION_CATEGORY_TO_VALUE = {
    'PROFESSION_NONE': 0,
    'PROFESSION_POLICE': 1,
    'PROFESSION_CONSTRUCTOR': 2,
}

CHILD_CATEGORY_TO_VALUE = {
    'ADULT': 0,
    'CHILD': 1,
}


def load_npy_files(input_path: str) -> List[np.ndarray]:
    """Load npy files from different input types."""
    input_path = Path(input_path)

    if input_path.is_dir():
        npy_files = sorted(input_path.glob('*.npy'))
        if not npy_files:
            raise ValueError(f'No .npy files found in directory: {input_path}')
        return [np.load(str(file_path), allow_pickle=True) for file_path in npy_files]

    if input_path.suffix == '.npy':
        return [np.load(str(input_path), allow_pickle=True)]

    if input_path.suffix == '.txt':
        with open(input_path, 'r', encoding='utf-8') as file_obj:
            npy_paths = [line.strip() for line in file_obj if line.strip()]
        if not npy_paths:
            raise ValueError(f'No valid paths found in text file: {input_path}')
        return [np.load(path, allow_pickle=True) for path in npy_paths]

    raise ValueError(
        f'Unsupported input type: {input_path}. Must be .npy file, directory, or .txt file'
    )


def load_review_json(json_path: str) -> Dict[str, dict]:
    """Load manual review JSON and index it by image_id."""
    with open(json_path, 'r', encoding='utf-8') as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, list):
        raise ValueError('Manual review JSON must contain a list of items')

    indexed = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get('image_id', '')).strip()
        if image_id:
            indexed[image_id] = item
    return indexed


def ensure_sample_length(sample: Iterable, expected_length: int = 14) -> List:
    """Convert sample to mutable list and pad missing columns."""
    values = list(sample)
    if len(values) < expected_length:
        values.extend([''] * (expected_length - len(values)))
    return values


def extract_json_categories(review_item: dict) -> Set[str]:
    """Extract category names directly from JSON and use them to overwrite npy labels."""
    json_categories: Set[str] = set()
    categories = review_item.get('categories', [])

    if not isinstance(categories, list):
        return json_categories

    for category_item in categories:
        if isinstance(category_item, dict):
            category_name = str(category_item.get('category', '')).strip()
            if category_name:
                json_categories.add(category_name)
        elif isinstance(category_item, str) and category_item.strip():
            json_categories.add(category_item.strip())

    return json_categories


def infer_gt_class(active_categories: Set[str], original_gt_class) -> int:
    """Infer gt_class from active categories, falling back to original where ambiguity remains."""
    main_matches = [
        category for category in MAIN_CATEGORY_PRIORITY if category in active_categories
    ]
    if main_matches:
        return MAIN_CATEGORY_TO_GT_CLASS[main_matches[0]]

    if any(category in active_categories for category in PROFESSION_CATEGORY_TO_VALUE):
        return 8

    if any(category in active_categories for category in SLOPE_CATEGORY_TO_VALUE):
        return 2

    if 'SV_SCHOOL_BUS' in active_categories or any(
        category in active_categories for category in STOP_SIGN_CATEGORY_TO_VALUE
    ):
        return 4

    if any(category in active_categories for category in CHILD_CATEGORY_TO_VALUE):
        if original_gt_class in [5, 6, 8]:
            return int(original_gt_class)
        return 8

    if active_categories:
        try:
            return int(original_gt_class)
        except (TypeError, ValueError):
            return -1

    return -1


def infer_sp_vehicle(active_categories: Set[str], original_sp_vehicle) -> int:
    """Infer sp_vehicle from active categories."""
    for category_name, value in SV_CATEGORY_TO_VALUE.items():
        if category_name in active_categories:
            return value

    for category_name, value in SLOPE_CATEGORY_TO_VALUE.items():
        if category_name in active_categories:
            return value

    if any(category in active_categories for category in STOP_SIGN_CATEGORY_TO_VALUE):
        return 6

    if 'CONSTRUCTION_VEHICLE' in active_categories and original_sp_vehicle == 9:
        return 9

    return -1


def infer_face_toward(active_categories: Set[str], original_face_toward):
    """Infer face_toward using values that subtype.py maps back consistently."""
    if 'FACE_TOWARD_FRONT' in active_categories:
        return 'front'

    if 'FACE_TOWARD_NO_FRONT' in active_categories:
        if original_face_toward not in ('', 'front', None):
            return original_face_toward
        return -1

    return ''


def infer_crossing(active_categories: Set[str], original_crossing):
    """Infer crossing field."""
    if 'CROSS' in active_categories:
        return 'crossing'

    if 'NO_CROSS' in active_categories:
        if original_crossing not in ('', 'crossing', None):
            return original_crossing
        return 'no_crossing'

    return ''


def update_sample_from_categories(sample: Iterable, active_categories: Set[str]) -> List:
    """Rewrite subtype-related label columns using the active category set."""
    updated_sample = ensure_sample_length(sample)

    original_gt_class = updated_sample[2]
    original_sp_vehicle = updated_sample[4]
    original_sub_type = updated_sample[3]
    original_face_toward = updated_sample[9]
    original_crossing = updated_sample[12]

    updated_sample[2] = infer_gt_class(active_categories, original_gt_class)
    updated_sample[3] = original_sub_type
    updated_sample[4] = infer_sp_vehicle(active_categories, original_sp_vehicle)

    profession_value = -1
    for category_name, value in PROFESSION_CATEGORY_TO_VALUE.items():
        if category_name in active_categories:
            profession_value = value
            break
    updated_sample[5] = profession_value

    door_positions = [
        position for category_name, position in DOOR_CATEGORY_TO_POSITION.items()
        if category_name in active_categories
    ]
    updated_sample[6] = 1 if door_positions else -1
    updated_sample[7] = door_positions if door_positions else []

    segment_values = {
        value for category_name, value in SEGMENT_CATEGORIES.items()
        if category_name in active_categories
    }
    updated_sample[8] = segment_values if segment_values else set()

    updated_sample[9] = infer_face_toward(active_categories, original_face_toward)

    robotaxi_value = -1
    for category_name, value in ROBOTAXI_CATEGORY_TO_VALUE.items():
        if category_name in active_categories:
            robotaxi_value = value
            break
    updated_sample[10] = robotaxi_value

    child_value = -1
    for category_name, value in CHILD_CATEGORY_TO_VALUE.items():
        if category_name in active_categories:
            child_value = value
            break
    updated_sample[11] = child_value

    updated_sample[12] = infer_crossing(active_categories, original_crossing)

    stop_sign_value = -1
    for category_name, value in STOP_SIGN_CATEGORY_TO_VALUE.items():
        if category_name in active_categories:
            stop_sign_value = value
            break
    updated_sample[13] = stop_sign_value

    return updated_sample


def reverse_manual_json_to_npy(input_path: str, json_path: str, output_path: str) -> None:
    """Apply corrected manual-review JSON back onto original npy samples."""
    npy_data_list = load_npy_files(input_path)
    review_items = load_review_json(json_path)

    updated_samples = []
    matched_image_ids = set()

    for data in npy_data_list:
        for sample in data:
            sample_list = ensure_sample_length(sample)
            image_id = str(sample_list[0]) if sample_list else ''
            review_item = review_items.get(image_id)

            if review_item is None:
                updated_samples.append(sample_list)
                continue

            json_categories = extract_json_categories(review_item)
            updated_samples.append(update_sample_from_categories(sample_list, json_categories))
            matched_image_ids.add(image_id)

    unmatched_image_ids = sorted(set(review_items.keys()) - matched_image_ids)
    output_array = np.array(updated_samples, dtype=object)
    np.save(output_path, output_array, allow_pickle=True)

    print(f'Successfully wrote {len(output_array)} samples to {output_path}')
    print(f'Matched review items: {len(matched_image_ids)} / {len(review_items)}')
    if unmatched_image_ids:
        print(f'Unmatched review items: {len(unmatched_image_ids)}')
        preview = ', '.join(unmatched_image_ids[:10])
        if preview:
            print(f'First unmatched ids: {preview}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert corrected subtype manual-review JSON back to .npy samples'
    )
    parser.add_argument(
        'input_path',
        help='Original input path (.npy file, directory of .npy files, or .txt file with .npy paths)'
    )
    parser.add_argument(
        'review_json',
        help='Corrected manual-review JSON file generated from subtype.py --manual'
    )
    parser.add_argument(
        'output_file',
        help='Path to output .npy file'
    )

    args = parser.parse_args()
    reverse_manual_json_to_npy(args.input_path, args.review_json, args.output_file)
