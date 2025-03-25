import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def shrink_polygon(pts, shrink_pixels=2):
    center = np.mean(pts, axis=0)
    new_pts = []

    for p in pts:
        direction = p - center
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = (direction / norm) * shrink_pixels
        new_pts.append(p - direction)

    return np.array(new_pts, dtype=np.float32)


def four_point_transform(image, pts):
    rect = order_points(pts)
    rect = shrink_polygon(rect, shrink_pixels=2)
    dst = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (256, 256), flags=cv2.INTER_CUBIC)


def perspective(image):
    DEBUG = False
    PRESET_POINTS = np.array(
        [[235, 6], [10, 16], [31, 243], [250, 233]], dtype=np.int32)

    gamma_img = cv2.LUT(
        image,
        np.array(
            [((i / 255.0) ** (1/0.9) * 255) for i in np.arange(256)],
            dtype=np.uint8
        )
    )

    gray = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    v = np.median(equalized)
    lower = int(max(0, (1.0 - 0.3) * v))
    upper = int(min(255, (1.0 + 0.3) * v))
    edged = cv2.Canny(equalized, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contour detection
    contours, _ = cv2.findContours(
        edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    screenCnt = None
    margin = 40
    detected_points = None

    # Vertex validation
    for idx, c in enumerate(contours):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if len(approx) == 4:
            points = approx.squeeze().astype(np.int32)
            edge_count = sum(
                (p[0] <= margin or p[0] >= 256 - margin or
                 p[1] <= margin or p[1] >= 256 - margin)
                for p in points
            )

            if edge_count >= 2:
                detected_points = points
                screenCnt = approx
                break

    # Coordinate verification
    USE_PRESET = False
    if detected_points is not None:
        ordered_detected = order_points(detected_points.astype(float))
        ordered_preset = order_points(PRESET_POINTS.astype(float))
        total_diff = np.sum(np.linalg.norm(
            ordered_detected - ordered_preset, axis=1))

        if total_diff > 40:
            USE_PRESET = True
    else:
        USE_PRESET = True

    # Perform transformation
    try:
        if USE_PRESET:
            result = four_point_transform(
                image, order_points(PRESET_POINTS.astype(float)))
        else:
            result = four_point_transform(image, detected_points)

        result = cv2.filter2D(result, -1, np.array([
            [-0.3, -0.3, -0.3],
            [-0.3,  3.4, -0.3],
            [-0.3, -0.3, -0.3]
        ]))

        return result

    except Exception as e:
        return image
