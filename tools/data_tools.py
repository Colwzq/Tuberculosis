def extract_specific_year_data(input_file_path, target_year, output_file_path):
    with open(input_file_path, "r") as f:
        lines = f.readlines()
    with open(output_file_path, "w") as f:
        for line in lines:
            file_name = line.strip().split()[0]
            if str(target_year) in file_name:
                f.write(line)


if __name__ == "__main__":
    input_file_path = "work_dirs/symformer_retinanet_p2t_cls_jail/result/cls_result.txt"
    output_file_path = (
        "work_dirs/symformer_retinanet_p2t_cls_jail/result/cls_result_2025.txt"
    )
    target_years = 2025
    extract_specific_year_data(input_file_path, target_years, output_file_path)
