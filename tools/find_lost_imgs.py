import os
import json

with open("file_list.json", "r") as f:
    file_list = json.load(f)

current_file = {
    file: file
    for file in os.listdir("/home/colwzq/data/medical_datasets/TBX11K/imgs/sick")
}
lost_file = []
for file in file_list:
    if file not in current_file:
        lost_file.append(file)
lost_file.sort()
print(lost_file)
