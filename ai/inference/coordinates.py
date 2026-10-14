"""Validate YOLO pixel boxes relative to the oriented original image."""
import math
from numbers import Real


class CoordinateError(ValueError):
    pass


def validate_image_size(image_size, result_shape):
    if len(image_size) != 2 or len(result_shape) != 2:
        raise CoordinateError('Expected width/height and height/width pairs')
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           or v <= 0 or int(v) != v for v in (*image_size, *result_shape)):
        raise CoordinateError('Dimensions must be positive finite integers')
    width, height = image_size
    if tuple(result_shape) != (height, width):
        raise CoordinateError('Coordinates are not relative to the oriented original image')
    return width, height


def normalize_box(coords, image_size):
    width, height = validate_image_size(image_size, tuple(reversed(image_size)))
    if len(coords) != 4 or any(isinstance(v, bool) or not isinstance(v, Real)
                              or not math.isfinite(v) for v in coords):
        raise CoordinateError('Expected four finite numeric coordinates')
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise CoordinateError('Box is inverted, empty or outside the original image')
    return x1 / width, y1 / height, x2 / width, y2 / height
