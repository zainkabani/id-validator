from cv2 import Mat
import cv2
from typing import Any, List, Optional, Set, Tuple
import numpy as np
import random


class PipelineStep:
    def __init__(self, fn_name: str, args: Tuple[Any, ...]):
        self.fn_name = fn_name
        self.args = args

    def __eq__(self, other):
        return self.fn_name == other.fn_name and self.args == other.args

    def __hash__(self):
        return hash((self.fn_name, self.args))

    def __repr__(self):
        return f"({self.fn_name}, ({self.args}))"

    def __str__(self):
        return f"({self.fn_name}, ({self.args}))"

    def __call__(self, img: Mat) -> Mat:
        return globals()[self.fn_name](img, *self.args)


class Pipeline:
    def __init__(self, steps: Tuple[PipelineStep, ...]):
        self.steps = steps

    def __repr__(self) -> str:
        return f"Pipeline({self.steps})"

    def __str__(self) -> str:
        return f"Pipeline({self.steps})"

    def execute(self, img: Mat) -> Mat:
        for step in self.steps:
            img = step(img)

        return img


def generate_pipelines() -> List[Pipeline]:
    pipelines_set: Set[Tuple[PipelineStep, ...]] = set()

    # Grayscale
    pipelines_set.add((PipelineStep(bw_img.__name__, ()),))

    # Invert
    new_pipelines = [
        pipeline + (PipelineStep(invert_img.__name__, ()),) for pipeline in pipelines_set]
    pipelines_set.update(new_pipelines)  # type: ignore

    # Threshold
    threshold_vals = [60, 80, 100, 120]
    new_pipelines = [pipeline + (PipelineStep(threshold_img.__name__, (threshold_val,)),)
                     for pipeline in pipelines_set for threshold_val in threshold_vals]
    # Want to add thresholding to all pipelines
    pipelines_set = set(new_pipelines)  # type: ignore

    # Denoise
    denoising_h_values = [11, 15, 19]
    denoising_search_window_sizes = [17, 21, 25]
    denoising_template_window_sizes = [5, 7]
    new_pipelines = [pipeline + (PipelineStep(denoise_img.__name__, (h, s, t)),)
                     for pipeline in pipelines_set for h in denoising_h_values for s in denoising_search_window_sizes for t in denoising_template_window_sizes]
    pipelines_set.update(new_pipelines)  # type: ignore

    # Erode or dilate
    kernel_vals = [5]
    erode_dilate_fns = [erode_img.__name__, dilate_img.__name__]
    new_pipelines = [pipeline + (PipelineStep(e_d_fn, (kernel_val,)),)
                     for pipeline in pipelines_set for e_d_fn in erode_dilate_fns for kernel_val in kernel_vals]
    pipelines_set.update(new_pipelines)  # type: ignore

    # Convert to list and shuffle
    final_pipelines = [Pipeline(pipeline) for pipeline in pipelines_set]
    random.shuffle(final_pipelines)

    return final_pipelines


def threshold_img(img: Mat, threshold: Optional[int]) -> Mat:
    if threshold is None:
        return img
    img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
    return img_thresh


def invert_img(img: Mat) -> Mat:
    return cv2.bitwise_not(img)


def bw_img(img: Mat) -> Mat:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise_img(img: Mat, h_value: int, search_window_value: int, template_window_value: int) -> Mat:
    denoised_img = cv2.fastNlMeansDenoising(
        img, None, h=h_value, searchWindowSize=search_window_value, templateWindowSize=template_window_value)  # type: ignore
    return denoised_img


def dilate_img(img: Mat, kernel_size: int) -> Mat:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def erode_img(img: Mat, kernel_size: int) -> Mat:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=1)


if __name__ == "__main__":
    thing = generate_pipelines()
    print(len(thing))
    print(thing[:5])
    resized_image = "images/7.jpg"
    img = cv2.imread(resized_image)
    thing[0].execute(img)
