from datetime import datetime
import os
import glob
import multiprocessing
from typing import Dict, List, Tuple


from pipeline import generate_pipelines
from validator import ValidationStatus, Validator, validate_async


def parse_input_file(path: str) -> Tuple[str, datetime]:
    with open(path, "r") as f:
        lines = f.readlines()
        name = lines[0].strip().lower()
        dob = datetime.strptime(lines[1].strip(), "%Y/%m/%d")
        return name, dob


if __name__ == '__main__':
    pipelines = generate_pipelines()

    manager = multiprocessing.Manager()
    validators = manager.list()

    print(len(pipelines))

    pool = multiprocessing.Pool()

    image_paths = glob.glob(f"images/*.jpg")
    # image_paths = [f"images/15.jpg"]
    print("Total images:", len(image_paths))
    for image_path in image_paths:
        base_file_name = image_path.split(os.sep)[-1].split(".")[0]

        input_file_path = f"input_data/{base_file_name}.txt"

        name, dob = parse_input_file(input_file_path)

        validator = Validator(
            pipelines[:50], image_path, base_file_name, name, dob)
        pool.apply_async(validate_async, args=(
            validator,), callback=validators.append)
    pool.close()
    pool.join()

    validation_statuses: Dict[ValidationStatus, List[str]] = {}

    for validator in validators:
        current_status = validator.get_status()
        validation_statuses[current_status] = validation_statuses.get(
            current_status, []) + [validator.file_base_name]

    for status, file_names in validation_statuses.items():
        for file_name in file_names:
            print(f"{status} {file_name}")
