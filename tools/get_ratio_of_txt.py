import numpy as np

RESULT_TXT_PATH = (
    "work_dirs/symformer_retinanet_p2t_cls_jail/result/cls_result_2025.txt"
)
OUTPUT_TXT_PATH = (
    "work_dirs/symformer_retinanet_p2t_cls_jail/result/cls_output_2025.txt"
)
# RESULT_TXT_PATH = "work_dirs/symformer_retinanet_p2t_cls/result/cls_result.txt"
results = open(RESULT_TXT_PATH, "r").readlines()
a = []
for res in results:
    res = res.strip().split()
    res = [float(x) for x in res[1:]]
    a.append(res)
a = np.array(a)
# a = np.loadtxt(RESULT_TXT_PATH)
max_index = np.argmax(a, axis=1)
print(sum(max_index == 0) / len(a))
print(sum(max_index == 1) / len(a))
print(sum(max_index == 2) / len(a))
with open(OUTPUT_TXT_PATH, "w") as f:
    label = ["healthy", "sick but not tb", "tb"]
    for i, index in enumerate(max_index):
        name = f"{results[i].strip().split()[0]} {label[index]}\n"
        f.write(name)
