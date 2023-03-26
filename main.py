from datetime import datetime
import os
import glob
import multiprocessing
from typing import Dict, List, Tuple


from pipeline import generate_pipelines
from validator import Validator, validate_async


def parse_input_file(path: str) -> Tuple[str, datetime]:
    with open(path, "r") as f:
        lines = f.readlines()
        name = lines[0].strip().lower()
        dob = datetime.strptime(lines[1].strip(), "%Y/%m/%d")
        return name, dob


if __name__ == '__main__':
    pipelines = generate_pipelines()
    print(f"Generated {len(pipelines)} image pre-processing pipelines")

    manager = multiprocessing.Manager()
    validators = manager.list()
    pool = multiprocessing.Pool()

    validation_statuses: Dict[bool, List[str]] = {}

    data_paths = os.listdir("data")
    # data_paths = ["1"]

    print("Total images:", len(data_paths))
    for data_path in data_paths:
        base_name = data_path.split(os.sep)[-1]

        try:
            id_path = glob.glob(f"data/{data_path}/id.*")[0]
        except IndexError:
            print(f"Skipping {data_path} (no id)")
            validation_statuses[False] = validation_statuses.get(
                False, []) + [data_path]
            continue

        try:
            headshot_path = glob.glob(f"data/{data_path}/headshot.*")[0]
        except IndexError:
            print(f"Skipping {data_path} (no headshot)")
            validation_statuses[False] = validation_statuses.get(
                False, []) + [data_path]
            continue

        try:
            info_path = glob.glob(f"data/{data_path}/info.txt")[0]
        except IndexError:
            print(f"Skipping {data_path} (no info)")
            validation_statuses[False] = validation_statuses.get(
                False, []) + [data_path]
            continue

        name, dob = parse_input_file(info_path)

        validator = Validator(
            pipelines[:50], base_name, id_path, headshot_path, name, dob)
        # validator = validate_async(validator)
        pool.apply_async(validate_async, args=(
            validator,), callback=validators.append)
    pool.close()
    pool.join()

    for validator in validators:
        current_status = validator.is_valid()
        validation_statuses[current_status] = validation_statuses.get(
            current_status, []) + [validator.base_name]

    print(f"Valid: {validation_statuses.get(True, [])}")
    print(f"Invalid: {validation_statuses.get(False, [])}")
