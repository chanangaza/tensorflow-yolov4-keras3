"""
MIT License

Copyright (c) 2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import List

import numpy as np

# pylint:disable=unused-argument,multiple-statements

def convert_dataset_to_ground_truth(
    dataset: np.ndarray, metayolos: np.ndarray, anchors: np.ndarray
) -> List[np.ndarray]: ...
def get_yolo_detections(
    yolo_0: np.ndarray,
    yolo_1: np.ndarray,
    yolo_2: np.ndarray,
    mask_0: np.ndarray,
    mask_1: np.ndarray,
    mask_2: np.ndarray,
    anchors: np.ndarray,
    beta_nms: float,
    new_coords: bool,
) -> np.ndarray: ...
def get_yolo_tiny_detections(
    yolo_0: np.ndarray,
    yolo_1: np.ndarray,
    mask_0: np.ndarray,
    mask_1: np.ndarray,
    anchors: np.ndarray,
    beta_nms: float,
    new_coords: bool,
) -> np.ndarray: ...
def fit_to_original(
    pred_bboxes: np.ndarray,
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
): ...
def yolo_tpu_layer(
    x: np.ndarray, logi: np.ndarray, num_masks: int, scale_x_y: float
): ...
