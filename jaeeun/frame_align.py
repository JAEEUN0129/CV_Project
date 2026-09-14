"""Align every frame of a clip to the model's 1024px face layout, steadily.

Used by ``hair_runner.py`` inside the isolated HairFastGAN environment, so like the runner
it imports nothing from the ``jaeeun`` package — only numpy, scipy, Pillow and dlib, which
that environment already has.

Aligning each frame on its own, as ``model.swap(..., align=True)`` does, causes two of the
jumps visible in restyled clips:

- dlib re-detects the 68 landmarks from scratch every frame, so the crop wobbles by a few
  pixels even when the subject holds still, and that wobble comes out of the generator as
  flicker. Here the landmarks are detected for the whole clip first and each point's path
  is low-pass filtered before cropping.
- A frame where no face is found raised, was skipped, and the survivors were renumbered —
  so a clip where the subject turns away lost half its frames and played back compressed.
  Here short gaps are interpolated from the frames on either side, and a frame that has
  no face for longer is reported as ``held`` so the caller can keep the timeline intact.

The crop geometry is transcribed from HairFastGAN's ``utils/shape_predictor.align_face``;
only its input (the landmarks) changes, so the crops stay in the layout the pretrained
encoders expect.
"""

from pathlib import Path

import dlib
import numpy as np
import PIL.Image
import scipy.ndimage

OUTPUT_SIZE = 1024
TRANSFORM_SIZE = 4096

DETECTED = "detected"
INTERPOLATED = "interpolated"
HELD = "held"


def load_detector(predictor_path: Path):
    """dlib face detector and 68-point predictor. The predictor ships with the weights."""
    if not Path(predictor_path).is_file():
        raise FileNotFoundError(f"landmark model missing: {predictor_path}")
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(str(predictor_path))


def detect_landmarks(image: PIL.Image.Image, detector, predictor) -> np.ndarray | None:
    """68x2 landmarks of the largest face, or None when no face is found."""
    pixels = np.array(image.convert("RGB"))
    faces = detector(pixels, 1)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda box: box.width() * box.height())
    shape = predictor(pixels, face)
    return np.array([[point.x, point.y] for point in shape.parts()], dtype=np.float64)


def fill_gaps(landmarks: list[np.ndarray | None], max_gap: int) -> tuple[list[np.ndarray | None], list[str]]:
    """Fill short runs of missing landmarks; mark longer ones as held.

    A run of missing frames between two detections, no longer than ``max_gap``, is filled
    by interpolating each coordinate linearly — detection usually fails there for a moment
    (motion blur, a hand passing) while the head keeps moving smoothly. A short run at the
    start or end copies the nearest detection. Anything longer means the face really is
    gone (the subject turned away), and inventing a face position would only feed the
    model a crop of the back of a head, so those frames are left for the caller to hold.
    """
    filled = list(landmarks)
    status = [DETECTED if lm is not None else HELD for lm in landmarks]
    found = [index for index, lm in enumerate(landmarks) if lm is not None]
    if not found or max_gap <= 0:
        return filled, status

    index = 0
    while index < len(landmarks):
        if landmarks[index] is not None:
            index += 1
            continue
        start = index
        while index < len(landmarks) and landmarks[index] is None:
            index += 1
        end = index  # first frame after the gap
        if end - start > max_gap:
            continue

        before = start - 1 if start > 0 else None
        after = end if end < len(landmarks) else None
        for frame in range(start, end):
            if before is not None and after is not None:
                weight = (frame - before) / (after - before)
                filled[frame] = (1 - weight) * landmarks[before] + weight * landmarks[after]
            else:
                filled[frame] = landmarks[before if before is not None else after].copy()
            status[frame] = INTERPOLATED
    return filled, status


def smooth_landmarks(landmarks: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Gaussian low-pass each landmark coordinate along time. ``landmarks`` is [T, 68, 2].

    Swept on a 24-frame clip with +/-16 degrees of head rotation, measuring flicker as the
    hair-region residual left after optical-flow motion compensation, and identity as
    ArcFace cosine similarity (to the neighbouring frame, to the first frame, worst frame):

        sigma  flicker  id_adjacent  id_to_first  id_worst
          0.0    4.045       0.9919       0.9735    0.9552
          0.5    3.589       0.9932       0.9711    0.9621
          0.8    3.098       0.9928       0.9600    0.9484
          1.5    3.422       0.9875       0.9443    0.9269
          2.5    4.487       0.9721       0.9163    0.8877

    Past about 1 the filter starts lagging the real head motion rather than only the
    detector's jitter: at 2.5 the crop trails the face so far that flicker is worse than
    not smoothing at all, and identity collapses. 0.5 is close to free — the best
    neighbouring and worst-frame identity of any setting, including none, for 11% less
    flicker. 0.8 halves the flicker further at about 1.4% identity.
    """
    if sigma <= 0 or len(landmarks) < 3:
        return landmarks
    # `nearest` keeps the first and last frames from being pulled toward zero.
    return scipy.ndimage.gaussian_filter1d(landmarks, sigma=sigma, axis=0, mode="nearest")


def quad_from_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, float]:
    """Oriented crop rectangle. Transcribed from HairFastGAN's align_face."""
    eye_left = landmarks[36:42].mean(axis=0)
    eye_right = landmarks[42:48].mean(axis=0)
    mouth_outer = landmarks[48:60]
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    eye_to_mouth = (mouth_outer[0] + mouth_outer[6]) * 0.5 - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    center = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([center - x - y, center - x + y, center + x + y, center + x - y])
    return quad, float(np.hypot(*x) * 2)


def crop_with_quad(image: PIL.Image.Image, quad: np.ndarray, qsize: float) -> PIL.Image.Image:
    """Shrink, crop, pad and warp to 1024px. Transcribed from HairFastGAN's align_face.

    One deliberate change: upstream shrinks with ``PIL.Image.ANTIALIAS``, which Pillow 10
    removed. It only runs for faces over about 2048px — rare in stills, routine in 4K
    video — so LANCZOS, the same filter under its current name, is used instead.
    """
    image = image.convert("RGB")
    quad = quad.copy()

    shrink = int(np.floor(qsize / OUTPUT_SIZE * 0.5))
    if shrink > 1:
        size = (int(np.rint(image.size[0] / shrink)), int(np.rint(image.size[1] / shrink)))
        image = image.resize(size, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(quad[:, 0].min())), int(np.floor(quad[:, 1].min())),
            int(np.ceil(quad[:, 0].max())), int(np.ceil(quad[:, 1].max())))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, image.size[0]), min(crop[3] + border, image.size[1]))
    if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
        image = image.crop(crop)
        quad -= crop[0:2]

    pad = (int(np.floor(quad[:, 0].min())), int(np.floor(quad[:, 1].min())),
           int(np.ceil(quad[:, 0].max())), int(np.ceil(quad[:, 1].max())))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
           max(pad[2] - image.size[0] + border, 0), max(pad[3] - image.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        pixels = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        height, width, _ = pixels.shape
        yy, xx, _ = np.ogrid[:height, :width, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(xx) / pad[0], np.float32(width - 1 - xx) / pad[2]),
            1.0 - np.minimum(np.float32(yy) / pad[1], np.float32(height - 1 - yy) / pad[3]),
        )
        blur = qsize * 0.02
        pixels += (scipy.ndimage.gaussian_filter(pixels, [blur, blur, 0]) - pixels) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        pixels += (np.median(pixels, axis=(0, 1)) - pixels) * np.clip(mask, 0.0, 1.0)
        image = PIL.Image.fromarray(np.uint8(np.clip(np.rint(pixels), 0, 255)), "RGB")
        quad += pad[:2]

    image = image.transform((TRANSFORM_SIZE, TRANSFORM_SIZE), PIL.Image.QUAD,
                            (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    return image.resize((OUTPUT_SIZE, OUTPUT_SIZE), PIL.Image.LANCZOS)


def align_one(image: PIL.Image.Image, detector, predictor) -> PIL.Image.Image:
    """Align a single photo, e.g. a shape or colour reference. Raises if no face is found."""
    landmarks = detect_landmarks(image, detector, predictor)
    if landmarks is None:
        raise ValueError("no face found")
    return crop_with_quad(image, *quad_from_landmarks(landmarks))


def align_clip(
    images: list[PIL.Image.Image],
    detector,
    predictor,
    sigma: float = 0.5,
    max_gap: int = 6,
) -> list[tuple[PIL.Image.Image | None, np.ndarray | None, str]]:
    """Align a clip. Returns one (crop, quad, status) per frame, in order.

    ``crop`` and ``quad`` are None for ``held`` frames. Smoothing runs separately over each
    stretch of consecutive frames that have landmarks, never across a held gap: the face
    positions on either side of a long absence are unrelated, and filtering across them
    would drag both toward a point where no face was.
    """
    raw = [detect_landmarks(image, detector, predictor) for image in images]
    filled, status = fill_gaps(raw, max_gap)

    index = 0
    while index < len(filled):
        if filled[index] is None:
            index += 1
            continue
        start = index
        while index < len(filled) and filled[index] is not None:
            index += 1
        smoothed = smooth_landmarks(np.stack(filled[start:index]), sigma)
        for offset, landmarks in enumerate(smoothed):
            filled[start + offset] = landmarks

    results = []
    for image, landmarks, state in zip(images, filled, status):
        if landmarks is None:
            results.append((None, None, state))
            continue
        quad, qsize = quad_from_landmarks(landmarks)
        results.append((crop_with_quad(image, quad, qsize), quad, state))
    return results
