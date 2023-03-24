from datetime import datetime
import os
import glob
import multiprocessing
from typing import Tuple

from PIL import Image
import cv2

from pipeline import generate_pipelines
from validator import Validator


RESIZED_IMAGE_PATH = "resized_images"


def resize_img(path: str):
    file_name = path.split(os.sep)[-1]

    already_resized = [resized_img.split(
        "/")[-1].split(".")[0] for resized_img in glob.glob(f"{RESIZED_IMAGE_PATH}/*")]

    if str(file_name.split(".")[0]) in already_resized:
        return

    print(f"Resizing {file_name}...")

    og_img = Image.open(path)
    og_img.save(os.path.join(RESIZED_IMAGE_PATH,
                file_name.split(".")[0] + ".png"), dpi=(300, 300))


def parse_input_file(path: str) -> Tuple[str, datetime]:
    with open(path, "r") as f:
        lines = f.readlines()
        name = lines[0].strip().lower()
        dob = datetime.strptime(lines[1].strip(), "%Y/%m/%d")
        return name, dob


if __name__ == '__main__':

    image_paths = glob.glob("images/*")
    for image_path in image_paths:
        resize_img(image_path)

    pipelines = generate_pipelines()

    # manager = Manager()
    # pipeline_total_success = manager.dict()
    # pipeline_failure = manager.dict()
    # scores = manager.list()

    print(len(pipelines))

    pool = multiprocessing.Pool(4)

    resized_images = glob.glob(f"{RESIZED_IMAGE_PATH}/*.png")
    resized_images = [f"{RESIZED_IMAGE_PATH}/6.png"]
    for resized_image in resized_images:
        base_file_name = resized_image.split(os.sep)[-1].split(".")[0]

        input_file_path = f"input_data/{base_file_name}.txt"

        # if not os.path.exists(input_file_path):
        #     continue

        name, dob = parse_input_file(input_file_path)

        img = cv2.imread(resized_image)
        validator = Validator(
            pipelines[:min(100, len(pipelines))], img, base_file_name, name, dob)
        # validator.validate()
        pool.apply_async(validator.validate, args=())
    pool.close()
    pool.join()
