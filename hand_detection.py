import cv2
import time
import os
import urllib.request
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hand_landmarker.task"
)


# ============================================================
# MEDIAPIPE MODEL URL
# ============================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)


# ============================================================
# DOWNLOAD MODEL IF IT DOES NOT EXIST
# ============================================================

def download_model():

    if os.path.exists(MODEL_PATH):

        print(
            "Hand Landmarker model already exists."
        )

        return

    print("=" * 60)

    print(
        "MediaPipe Hand Landmarker model not found."
    )

    print(
        "Downloading model..."
    )

    print("=" * 60)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    try:

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print(
            "\nModel downloaded successfully."
        )

    except Exception as e:

        print(
            "\nERROR: Could not download MediaPipe model."
        )

        print(
            "Error:",
            e
        )

        print(
            "\nPlease check your internet connection "
            "and run the program again."
        )

        exit()


# ============================================================
# DOWNLOAD MODEL
# ============================================================

download_model()


# ============================================================
# MEDIAPIPE HAND LANDMARKER OPTIONS
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.HandLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5

)


# ============================================================
# CREATE HAND LANDMARKER
# ============================================================

detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("=" * 60)

    print(
        "ERROR: Could not open webcam."
    )

    print("=" * 60)

    print(
        "Please check whether your webcam is connected "
        "or being used by another application."
    )

    detector.close()

    exit()


# ============================================================
# CAMERA SETTINGS
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# FPS VARIABLES
# ============================================================

previous_time = 0

frame_timestamp_ms = 0


# ============================================================
# HAND LANDMARK CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17)

]


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand_landmarks(
    frame,
    hand_landmarks
):

    height, width, _ = frame.shape

    points = []


    # --------------------------------------------------------
    # CONVERT NORMALIZED COORDINATES TO PIXELS
    # --------------------------------------------------------

    for landmark in hand_landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )


    # --------------------------------------------------------
    # DRAW CONNECTIONS
    # --------------------------------------------------------

    for start_index, end_index in HAND_CONNECTIONS:

        start_point = points[
            start_index
        ]

        end_point = points[
            end_index
        ]

        cv2.line(

            frame,

            start_point,

            end_point,

            (0, 255, 0),

            2

        )


    # --------------------------------------------------------
    # DRAW LANDMARK POINTS
    # --------------------------------------------------------

    for point in points:

        cv2.circle(

            frame,

            point,

            5,

            (0, 0, 255),

            -1

        )


# ============================================================
# START PROGRAM
# ============================================================

print("=" * 60)

print(
    "SIGN LANGUAGE RECOGNITION"
)

print(
    "REAL-TIME HAND LANDMARK DETECTION"
)

print("=" * 60)

print(
    "Webcam started successfully."
)

print(
    "Show your hand in front of the camera."
)

print(
    "Press 'Q' to quit."
)

print("=" * 60)


# ============================================================
# MAIN LOOP
# ============================================================

while True:


    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = cap.read()


    if not success:

        print(
            "ERROR: Failed to read webcam frame."
        )

        break


    # --------------------------------------------------------
    # FLIP FRAME
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # CONVERT BGR TO RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    # --------------------------------------------------------
    # CREATE MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame

    )


    # --------------------------------------------------------
    # UPDATE TIMESTAMP
    # --------------------------------------------------------

    frame_timestamp_ms += 33


    # --------------------------------------------------------
    # DETECT HANDS
    # --------------------------------------------------------

    result = detector.detect_for_video(

        mp_image,

        frame_timestamp_ms

    )


    # --------------------------------------------------------
    # CHECK HAND DETECTION
    # --------------------------------------------------------

    hand_detected = (

        len(
            result.hand_landmarks
        ) > 0

    )


    # --------------------------------------------------------
    # DRAW DETECTED HANDS
    # --------------------------------------------------------

    if hand_detected:

        for hand_landmarks in result.hand_landmarks:

            draw_hand_landmarks(

                frame,

                hand_landmarks

            )


    # --------------------------------------------------------
    # FPS CALCULATION
    # --------------------------------------------------------

    current_time = time.time()


    if previous_time != 0:

        fps = (

            1 /
            (
                current_time -
                previous_time
            )

        )

    else:

        fps = 0


    previous_time = current_time


    # --------------------------------------------------------
    # DETECTION STATUS
    # --------------------------------------------------------

    if hand_detected:

        status_text = (
            "HAND DETECTED"
        )

        status_color = (
            0,
            255,
            0
        )

    else:

        status_text = (
            "NO HAND DETECTED"
        )

        status_color = (
            0,
            0,
            255
        )


    # --------------------------------------------------------
    # STATUS BOX
    # --------------------------------------------------------

    cv2.rectangle(

        frame,

        (20, 20),

        (420, 110),

        (30, 30, 30),

        -1

    )


    # --------------------------------------------------------
    # DISPLAY STATUS
    # --------------------------------------------------------

    cv2.putText(

        frame,

        status_text,

        (40, 55),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        status_color,

        2,

        cv2.LINE_AA

    )


    # --------------------------------------------------------
    # DISPLAY FPS
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"FPS: {int(fps)}",

        (40, 90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    # --------------------------------------------------------
    # DISPLAY INSTRUCTIONS
    # --------------------------------------------------------

    cv2.putText(

        frame,

        "Press Q to Quit",

        (
            frame.shape[1] - 230,
            40
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    # --------------------------------------------------------
    # SHOW FRAME
    # --------------------------------------------------------

    cv2.imshow(

        "Sign Language Recognition - Hand Detection",

        frame

    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = (

        cv2.waitKey(1)
        &
        0xFF

    )


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()


print(
    "\nWebcam closed successfully."
)

print(
    "Hand detection program finished."
)