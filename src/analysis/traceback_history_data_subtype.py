import os
import json
import argparse

from glob import glob
from tqdm import tqdm
from typing import List, Any
from collections import defaultdict

TARGET_CAT = [
    "SV_POLICE_CAR",
    "SV_FIRE_ENGINE",
    "SV_AMBULANCE"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", "-t", type=str, help="target json file")
    parser.add_argument("--reference-dir", "-r", type=str, help="reference image dir")
    parser.add_argument("--output", "-o", type=str, help="output txt path")
    return parser.parse_args()

def filter_single(single_data: Any):
    evoloved_catgories = [(c['category'] in TARGET_CAT) and (c['decision'] == 'reject') \
            for c in single_data["categories"]]
    return any(evoloved_catgories)

def filter_all(origin_data: List[Any]):
    total_num = len(origin_data)
    results = []
    for data in tqdm(origin_data, total=total_num):
        if (filter_single(data)):
            results.append(data)
    return results

def get_image_id(all_data):
    origin_ids = [it['image_id'].split('/')[-1] for it in all_data]
    result_ids = []
    for id_ in origin_ids:
        new_id = '_'.join(id_.split('_')[:4])
        result_ids.append(new_id)
    return result_ids

def parse_txt_file(file: str):
    image_ids = []
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            _, cnt, _ = line.split('\t')
            cnt = json.loads(cnt)
            image_ids.append(cnt['fileName'])
    return image_ids

def img_id_to_task_id(save_dir: str):
    task_ids = os.listdir(save_dir)
    img_id_task_id_map = dict()
    for task_id in tqdm(task_ids, total=len(task_ids)):
        task_ids_index_file = os.path.join(save_dir, task_id, "result.txt")
        parse_result = parse_txt_file(task_ids_index_file)
        for image_id in parse_result:
            im_id = image_id.split('.')[0]
            img_id_task_id_map[im_id] = task_id
    return img_id_task_id_map

def main(args: argparse.Namespace):
    with open(args.target) as f:
        target_json_cnt = json.load(f)
    filter_result = filter_all(target_json_cnt)
    with open(args.output.replace('.txt', '.json'), 'w') as f:
        json.dump(filter_result, f)
    filter_result_image_ids = get_image_id(filter_result)
    img_task_map = img_id_to_task_id(args.reference_dir)

    final_task_ids = set()
    for im in filter_result_image_ids:
        if im in img_task_map:
            final_task_ids.add(img_task_map[im])
    print(f"Return {len(final_task_ids)} cases.")

    with open(args.output, "w") as f:
        for img_id in final_task_ids:
            f.write(img_id + "\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)

