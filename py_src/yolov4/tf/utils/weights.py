"""
MIT License

Copyright (c) 2020-2021 Hyeonki Hong <hhk7734@gmail.com>

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
import numpy as np


def load_weights(model, weights_file: str):
    with open(weights_file, "rb") as fd:
        # major, minor, revision, seen, _
        _np_fromfile(fd, dtype=np.int32, count=5)

        for layer in model.layers:
            if "convolutional" in layer.name:
                if not yolo_conv2d_load_weights(layer, fd):
                    break

        if len(fd.read()) != 0:
            raise ValueError("Model and weights file do not match.")


def _np_fromfile(fd, dtype, count: int):
    data = np.fromfile(fd, dtype=dtype, count=count)
    if len(data) != count:
        if len(data) == 0:
            return None
        raise ValueError("Model and weights file do not match.")
    return data


def yolo_conv2d_load_weights(yolo_conv2d, fd) -> bool:
    conv2d = None
    batch_normalization = None
    for layer in yolo_conv2d.layers:
        if "batch_normalization" in layer.name:
            batch_normalization = layer
        elif "conv2d" in layer.name:
            conv2d = layer

    filters = conv2d.filters

    if batch_normalization is not None:
        # darknet weights: [beta, gamma, mean, variance]
        bn_weights = _np_fromfile(fd, dtype=np.float32, count=4 * filters)
        if bn_weights is None:
            return False
        # tf weights: [gamma, beta, mean, variance]
        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

        batch_normalization.set_weights(bn_weights)

    conv_bias = None
    if conv2d.use_bias:
        conv_bias = _np_fromfile(fd, dtype=np.float32, count=filters)
        if conv_bias is None:
            return False

    # darknet shape (out_dim, in_dim, kernel_size, kernel_size)
    conv_shape = (filters, conv2d.input.shape[-1], *conv2d.kernel_size)

    conv_weights = _np_fromfile(
        fd, dtype=np.float32, count=np.product(conv_shape)
    )
    if conv_weights is None:
        return False
    # tf shape (kernel_size, kernel_size, in_dim, out_dim)
    conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

    if conv_bias is None:
        conv2d.set_weights([conv_weights])
    else:
        conv2d.set_weights([conv_weights, conv_bias])

    return True


def save_weights(model, weights_file: str, to: str = ""):
    with open(weights_file, "wb") as fd:
        # major, minor, revision, seen, _
        np.array([0, 2, 5, 32032000, 0], dtype=np.int32).tofile(fd)

        for layer in model.layers:
            if "convolutional" in layer.name:
                yolo_conv2d_save_weights(layer, fd)
                if layer.name == to:
                    break


def yolo_conv2d_save_weights(yolo_conv2d, fd):
    conv2d = None
    batch_normalization = None
    for layer in yolo_conv2d.layers:
        if "batch_normalization" in layer.name:
            batch_normalization = layer
        elif "conv2d" in layer.name:
            conv2d = layer

    if batch_normalization is not None:
        # tf weights: [gamma, beta, mean, variance]
        bn_weights = np.stack(batch_normalization.get_weights())
        # darknet weights: [beta, gamma, mean, variance]
        bn_weights[[1, 0, 2, 3]].reshape((-1,)).tofile(fd)

    # tf shape (height, width, in_dim, out_dim)
    if conv2d.use_bias:
        conv_weights, conv_bias = conv2d.get_weights()
        conv_bias.tofile(fd)
    else:
        conv_weights = conv2d.get_weights()[0]

    # darknet shape (out_dim, in_dim, height, width)
    conv_weights.transpose([3, 2, 0, 1]).reshape((-1,)).tofile(fd)
