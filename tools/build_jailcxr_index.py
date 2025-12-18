import pydicom
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
from collections import deque
import shutil
import json


def flatten_folder_bfs(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    queue = deque([source_dir])
    folder_name_set = set()
    folder_name_set.add(source_dir)
    while queue:
        current_path = queue.popleft()
        if current_path == target_dir:
            continue
        # remove macOS system files
        if ".DS_Store" in current_path or "._.DS_Store" in current_path:
            continue
        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            if item_path in folder_name_set:
                raise Exception(f"Duplicate folder detected: {item_path}")
            else:
                folder_name_set.add(item_path)
            if os.path.isdir(item_path):
                queue.append(item_path)
            else:
                file_name = item_path.split("/")[-3] + ".dcm"
                target_file_path = os.path.join(target_dir, file_name)
                if os.path.exists(target_file_path):
                    # raise Exception(f"File already exists: {target_file_path}")
                    base, ext = os.path.splitext(file_name)
                    counter = 1
                    while os.path.exists(
                        os.path.join(target_dir, f"{base}_{counter}{ext}")
                    ):
                        counter += 1
                    target_file_path = os.path.join(
                        target_dir, f"{base}_{counter}{ext}"
                    )
                print(f"Copying {item_path} to {target_file_path}")
                shutil.copy(item_path, target_file_path)


if __name__ == "__main__":
    ROOT_PATH = "/home/colwzq/data/jail_cxr/"
    top_index = ["2023-肺部影像片", "2024-肺部影像片", "2025-肺部影像片"]
    for i, index_name in enumerate(top_index):
        dcm_file_top_path = ROOT_PATH + index_name
        flatten_folder_bfs(dcm_file_top_path, f"{ROOT_PATH}{2023+i}_cxr")

# 2023 4918
# 2024 3763
# 2025 8342
