def image_to_yolo(bbox: list[int], imsize) -> list[float]:
    """
    Converts the image bbox (x1, y1, x2, y2) to yolo bbox (xc, yc, w, h),
    where xc, yc, w, h \in [0; 1] (proportion to the image size)
    """

    x1, y1, x2, y2 = bbox

    w, h = float(x2 - x1) / imsize, float(y2 - y1) / imsize
    assert w >= 0.0 and w <= 1.0
    assert h >= 0.0 and h <= 1.0

    xc, yc = float(x1) / imsize + w / 2, float(y1) / imsize + h / 2
    assert xc >= 0.0 and xc <= 1.0
    assert yc >= 0.0 and yc <= 1.0

    return xc, yc, w, h

def yolo_to_image(bbox: list[float], imsize) -> list[int]:
    """
    Converts the yolo bbox (xc, yc, w, h) to image bbox (x1, y1, x2, y2).
    """

    xc, yc, w, h = bbox

    x1, y1 = int((xc - w / 2) * imsize), int((yc - h / 2) * imsize)
    assert x1 >= 0 and x1 <= imsize
    assert y1 >= 0 and y1 <= imsize

    weight, height = int(w * imsize), int(h * imsize)
    x2, y2 = x1 + weight, y1 + height
    assert x2 >= 0 and x2 <= imsize
    assert y2 >= 0 and y2 <= imsize

    return x1, y1, x2, y2

