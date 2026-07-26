# ============================================================
# REALTIME_PREDICTION.PY
# SIGN LANGUAGE RECOGNITION SYSTEM
# PHASE 13 - FINAL REAL-TIME RECOGNITION + PROFESSIONAL UI
#
# FINAL CLASSES:
# A-Z + del + space
#
# FEATURES:
# - MediaPipe Hand Landmarks
# - Random Forest Prediction
# - Prediction Confidence
# - Top 3 Predictions
# - Prediction Smoothing
# - Stable Gesture Detection
# - Duplicate Prediction Prevention
# - Space Gesture
# - Delete Gesture
# - Word Building
# - Autocomplete Suggestions
# - Sentence Building
# - Text-to-Speech
# - FPS Display
# - Professional Responsive UI
# - All Controls Visible
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import cv2
import json
import time
import joblib
import numpy as np
import mediapipe as mp
import pyttsx3

from collections import Counter, deque
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sign_language_model.pkl"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_labels.json"
)

HAND_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 50

SMOOTHING_FRAMES = 10

MAJORITY_THRESHOLD = 0.70

COOLDOWN_FRAMES = 15

MIN_STABLE_FRAMES = 10

REQUIRE_HAND_RESET = True

CAMERA_WIDTH = 1280

CAMERA_HEIGHT = 720


# ============================================================
# APPLICATION WINDOW
#
# IMPORTANT:
# The previous UI used 1650x900.
# This caused bottom controls to disappear on smaller screens.
#
# New compact layout:
# 1280 x 800
#
# Everything is designed to stay inside this area.
# ============================================================

WINDOW_NAME = (
    "Sign Language Recognition System"
)

CANVAS_WIDTH = 1280

CANVAS_HEIGHT = 800


# ============================================================
# PROFESSIONAL COLOR PALETTE
# OpenCV uses BGR format
# ============================================================

WHITE = (
    255,
    255,
    255
)

BLACK = (
    25,
    25,
    25
)

NAVY = (
    110,
    65,
    20
)

DARK_NAVY = (
    75,
    45,
    15
)

GREEN = (
    50,
    150,
    50
)

DARK_GREEN = (
    40,
    120,
    40
)

RED = (
    55,
    55,
    200
)

DARK_RED = (
    40,
    40,
    160
)

BLUE = (
    180,
    110,
    35
)

DARK_BLUE = (
    130,
    80,
    20
)

PURPLE = (
    130,
    80,
    130
)

ORANGE = (
    0,
    140,
    230
)

YELLOW = (
    0,
    190,
    230
)

LIGHT_BG = (
    245,
    247,
    249
)

PANEL_BG = (
    255,
    255,
    255
)

BORDER = (
    215,
    220,
    225
)

TEXT_DARK = (
    35,
    35,
    35
)

TEXT_GRAY = (
    100,
    100,
    100
)

LANDMARK_GREEN = (
    0,
    220,
    0
)


# ============================================================
# WORD LIST
# ============================================================

WORD_LIST = [

    "A",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",

    "BE",
    "BUT",
    "BY",
    "CAN",
    "COME",

    "COULD",
    "DID",
    "DO",
    "FOR",
    "FROM",

    "GET",
    "GO",
    "GOOD",
    "HAVE",
    "HE",

    "HELLO",
    "HELP",
    "HER",
    "HERE",
    "HEY",

    "HIM",
    "HIS",
    "HOW",
    "I",
    "IF",

    "IN",
    "IS",
    "IT",
    "ME",
    "MY",

    "NO",
    "NOT",
    "NOW",
    "OF",
    "ON",

    "ONE",
    "OR",
    "OUR",
    "PLEASE",

    "SHE",
    "SO",
    "THANK",
    "THANKS",

    "THAT",
    "THE",
    "THEY",
    "THIS",

    "TO",
    "TODAY",
    "TOMORROW",
    "UP",

    "US",
    "WAS",
    "WE",
    "WHAT",

    "WHEN",
    "WHERE",
    "WHO",
    "WHY",

    "WILL",
    "WITH",
    "YES",
    "YOU",
    "YOUR",

    "HI",
    "WELCOME",

    "GOODMORNING",
    "GOODAFTERNOON",
    "GOODEVENING",
    "GOODNIGHT",

    "NAME",
    "MYNAME",
    "FRIEND",
    "FAMILY",

    "HOME",
    "SCHOOL",
    "COLLEGE",

    "STUDENT",
    "TEACHER",
    "DOCTOR",

    "ENGINEER",
    "COMPUTER",
    "PROGRAM",

    "PROJECT",
    "WORK",
    "JOB",

    "FOOD",
    "WATER",
    "TEA",
    "COFFEE",

    "NEED",
    "WANT",
    "LIKE",
    "LOVE",

    "KNOW",
    "THINK",
    "UNDERSTAND",

    "LEARN",
    "READ",
    "WRITE",

    "SPEAK",
    "SIGN",
    "LANGUAGE",

    "GREAT",
    "BEST",
    "NICE",

    "HAPPY",
    "FINE",
    "OK",
    "OKAY",

    "NEVER",
    "BAD",
    "SORRY",

    "YESTERDAY",
    "MORNING",
    "AFTERNOON",

    "EVENING",
    "NIGHT",
    "DAY",

    "WEEK",
    "MONTH",
    "YEAR",

    "AI",
    "ML",
    "PYTHON",
    "CODE",

    "CODING",
    "DATA",
    "MODEL",

    "SOFTWARE",
    "DEVELOPER",
    "WEB",

    "APP",
    "APPLICATION",

    "ONE",
    "TWO",
    "THREE",
    "FOUR",

    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",

    "NINE",
    "TEN"
]


WORD_LIST = list(
    dict.fromkeys(
        WORD_LIST
    )
)


# ============================================================
# TEXT DRAWING FUNCTION
# ============================================================

def draw_text(
    frame,
    text,
    position,
    font_scale=0.7,
    color=TEXT_DARK,
    thickness=1
):

    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# PANEL FUNCTION
# ============================================================

def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    title,
    header_color=NAVY
):

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        PANEL_BG,
        -1
    )

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        BORDER,
        1
    )

    header_height = 38

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y1 + header_height
        ),
        header_color,
        -1
    )

    draw_text(
        frame,
        title,
        (
            x1 + 12,
            y1 + 26
        ),
        0.52,
        WHITE,
        2
    )


# ============================================================
# BUTTON FUNCTION
# ============================================================

def draw_button(
    frame,
    x1,
    y1,
    x2,
    y2,
    key,
    title,
    description,
    color
):

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        WHITE,
        -1
    )

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        BORDER,
        1
    )

    box_w = 48

    cv2.rectangle(
        frame,
        (
            x1 + 8,
            y1 + 8
        ),
        (
            x1 + 8 + box_w,
            y2 - 8
        ),
        color,
        -1
    )

    draw_text(
        frame,
        key,
        (
            x1 + 23,
            y1 + 43
        ),
        0.90,
        WHITE,
        2
    )

    draw_text(
        frame,
        title,
        (
            x1 + 68,
            y1 + 27
        ),
        0.46,
        color,
        2
    )

    draw_text(
        frame,
        description,
        (
            x1 + 68,
            y1 + 48
        ),
        0.35,
        TEXT_DARK,
        1
    )


# ============================================================
# PROGRESS BAR
# ============================================================

def draw_progress_bar(
    frame,
    x1,
    y1,
    x2,
    y2,
    percentage
):

    percentage = max(
        0,
        min(
            100,
            percentage
        )
    )

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (
            225,
            225,
            225
        ),
        -1
    )

    fill_width = int(
        (
            x2 - x1
        )
        *
        percentage
        /
        100
    )

    if fill_width > 0:

        cv2.rectangle(
            frame,
            (
                x1,
                y1
            ),
            (
                x1 + fill_width,
                y2
            ),
            GREEN,
            -1
        )

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        BORDER,
        1
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print(
    "=" * 70
)

print(
    "SIGN LANGUAGE RECOGNITION SYSTEM"
)

print(
    "PHASE 13 - REAL-TIME RECOGNITION"
)

print(
    "=" * 70
)


required_files = [

    MODEL_PATH,

    LABELS_PATH,

    HAND_MODEL_PATH

]


for required_file in required_files:

    if not os.path.exists(
        required_file
    ):

        print()

        print(
            "ERROR: Required file not found:"
        )

        print(
            required_file
        )

        exit()


print()

print(
    "All required files found."
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded:",
        type(model)
    )

    print(
        "Features:",
        model.n_features_in_
    )

except Exception as e:

    print(
        "ERROR loading model:",
        e
    )

    exit()


# ============================================================
# LOAD CLASS LABELS
# ============================================================

try:

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_labels = json.load(
            file
        )

    print(
        "Labels loaded:",
        len(class_labels)
    )

    print(
        "Classes:",
        class_labels
    )

except Exception as e:

    print(
        "ERROR loading labels:",
        e
    )

    exit()


# ============================================================
# TEXT TO SPEECH
# ============================================================

try:

    engine = pyttsx3.init()

    engine.setProperty(
        "rate",
        150
    )

    engine.setProperty(
        "volume",
        1.0
    )

except Exception as e:

    print(
        "TTS initialization failed:",
        e
    )

    engine = None


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

try:

    base_options = python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(

        base_options=base_options,

        running_mode=vision.RunningMode.VIDEO,

        num_hands=1,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5

    )

    detector = (
        vision.HandLandmarker
        .create_from_options(
            options
        )
    )

except Exception as e:

    print(
        "MediaPipe initialization failed:",
        e
    )

    exit()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    0
)


if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    detector.close()

    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


# ============================================================
# VARIABLES
# ============================================================

sentence = ""

current_word = ""

suggestions = []

prediction_history = deque(
    maxlen=SMOOTHING_FRAMES
)

last_accepted_prediction = None

current_prediction = "No Hand"

current_confidence = 0.0

top_predictions = []

cooldown_counter = 0

hand_detected = False

frame_timestamp_ms = 0

previous_time = time.time()

fps = 0.0


# ============================================================
# AUTOCOMPLETE
# ============================================================

def get_word_suggestions(
    current_word,
    max_results=3
):

    if not current_word:

        return []

    prefix = (
        current_word
        .strip()
        .upper()
    )

    if not prefix:

        return []

    matches = []

    for word in WORD_LIST:

        word = word.upper()

        if (

            word.startswith(
                prefix
            )

            and

            word != prefix

        ):

            matches.append(
                word
            )

    matches.sort(
        key=lambda word: (
            len(word),
            word
        )
    )

    return matches[
        :max_results
    ]


# ============================================================
# ACCEPT AUTOCOMPLETE SUGGESTION
# ============================================================

def accept_suggestion(
    suggestion
):

    global sentence

    global current_word

    global suggestions

    if not suggestion:

        return

    suggestion = (
        suggestion.upper()
    )

    if (

        sentence

        and

        not sentence.endswith(
            " "
        )

    ):

        sentence += " "

    sentence += suggestion

    sentence += " "

    current_word = ""

    suggestions = []


# ============================================================
# ADD LETTER
# ============================================================

def add_letter(
    letter
):

    global current_word

    global suggestions

    if len(letter) != 1:

        return

    if not letter.isalpha():

        return

    letter = (
        letter.upper()
    )

    current_word += letter

    suggestions = (
        get_word_suggestions(
            current_word
        )
    )


# ============================================================
# DELETE
# ============================================================

def delete_character():

    global current_word

    global sentence

    global suggestions

    if current_word:

        current_word = (
            current_word[:-1]
        )

        suggestions = (
            get_word_suggestions(
                current_word
            )
        )

        return

    if sentence:

        sentence = (
            sentence.rstrip()
        )

        if " " in sentence:

            sentence = (
                sentence.rsplit(
                    " ",
                    1
                )[0]
            )

        else:

            sentence = ""


# ============================================================
# SPEAK SENTENCE
# ============================================================

def speak_sentence():

    if engine is None:

        return

    text = (
        sentence.strip()
    )

    if current_word:

        if text:

            text += " "

        text += current_word

    if not text:

        return

    try:

        engine.say(
            text
        )

        engine.runAndWait()

    except Exception as e:

        print(
            "TTS error:",
            e
        )


# ============================================================
# CREATE WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    CANVAS_WIDTH,
    CANVAS_HEIGHT
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ CAMERA
    # ========================================================

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read webcam frame."
        )

        break


    # ========================================================
    # MIRROR CAMERA
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    frame_height, frame_width = (
        frame.shape[:2]
    )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    delta_time = (

        current_time

        -

        previous_time

    )

    previous_time = (
        current_time
    )

    if delta_time > 0:

        fps = (

            0.9
            *
            fps

            +

            0.1
            *
            (
                1
                /
                delta_time
            )

        )


    # ========================================================
    # MEDIAPIPE INPUT
    # ========================================================

    rgb_frame = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    mp_image = mp.Image(

        image_format=(
            mp.ImageFormat.SRGB
        ),

        data=rgb_frame

    )


    frame_timestamp_ms += 33


    result = (
        detector.detect_for_video(

            mp_image,

            frame_timestamp_ms

        )
    )


    # ========================================================
    # RESET PREDICTION DISPLAY
    # ========================================================

    current_prediction = (
        "No Hand"
    )

    current_confidence = 0.0

    top_predictions = []

    hand_detected = False


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand_detected = True

        hand_landmarks = (
            result.hand_landmarks[0]
        )


        if len(
            hand_landmarks
        ) == 21:

            wrist = (
                hand_landmarks[0]
            )

            features = []


            # =================================================
            # CREATE 63 FEATURES
            # =================================================

            for landmark in hand_landmarks:

                features.extend([

                    landmark.x
                    -
                    wrist.x,

                    landmark.y
                    -
                    wrist.y,

                    landmark.z
                    -
                    wrist.z

                ])


            features = np.array(

                features,

                dtype=np.float32

            ).reshape(

                1,

                -1

            )


            try:

                # =============================================
                # PREDICTION
                # =============================================

                prediction = (
                    model.predict(
                        features
                    )[0]
                )


                probabilities = (
                    model.predict_proba(
                        features
                    )[0]
                )


                current_prediction = (
                    str(
                        prediction
                    )
                )


                current_confidence = (

                    float(

                        np.max(
                            probabilities
                        )

                    )

                    *

                    100

                )


                # =============================================
                # TOP 3 PREDICTIONS
                # =============================================

                top_indices = (
                    np.argsort(
                        probabilities
                    )[::-1][:3]
                )


                for index in top_indices:

                    label = str(

                        model.classes_[

                            index

                        ]

                    )


                    probability = (

                        float(

                            probabilities[

                                index

                            ]

                        )

                        *

                        100

                    )


                    top_predictions.append(

                        (

                            label,

                            probability

                        )

                    )


                # =============================================
                # PREDICTION SMOOTHING
                # =============================================

                if (

                    current_confidence

                    >=

                    CONFIDENCE_THRESHOLD

                ):

                    prediction_history.append(

                        current_prediction

                    )

                else:

                    prediction_history.clear()


                # =============================================
                # STABLE PREDICTION
                # =============================================

                if len(

                    prediction_history

                ) >= MIN_STABLE_FRAMES:

                    most_common = (

                        Counter(

                            prediction_history

                        )

                        .most_common(

                            1

                        )[0]

                    )


                    stable_prediction = (

                        most_common[0]

                    )


                    stable_count = (

                        most_common[1]

                    )


                    stability = (

                        stable_count

                        /

                        len(

                            prediction_history

                        )

                    )


                    if (

                        stability

                        >=

                        MAJORITY_THRESHOLD

                    ):

                        if cooldown_counter == 0:

                            can_accept = True


                            # =================================
                            # PREVENT DUPLICATE PREDICTION
                            # =================================

                            if REQUIRE_HAND_RESET:

                                if (

                                    stable_prediction

                                    ==

                                    last_accepted_prediction

                                ):

                                    can_accept = False


                            if can_accept:

                                # =============================
                                # SPACE
                                # =============================

                                if (

                                    stable_prediction.lower()

                                    ==

                                    "space"

                                ):

                                    if current_word:

                                        if (

                                            sentence

                                            and

                                            not sentence.endswith(
                                                " "
                                            )

                                        ):

                                            sentence += " "


                                        sentence += (
                                            current_word
                                        )

                                        sentence += " "


                                        current_word = ""

                                        suggestions = []


                                    elif (

                                        sentence

                                        and

                                        not sentence.endswith(
                                            " "
                                        )

                                    ):

                                        sentence += " "


                                # =============================
                                # DELETE
                                # =============================

                                elif (

                                    stable_prediction.lower()

                                    ==

                                    "del"

                                ):

                                    delete_character()


                                # =============================
                                # LETTER A-Z
                                # =============================

                                elif (

                                    stable_prediction
                                    in
                                    class_labels

                                ):

                                    if (

                                        stable_prediction.lower()

                                        not in [

                                            "space",

                                            "del"

                                        ]

                                    ):

                                        add_letter(

                                            stable_prediction

                                        )


                                # =============================
                                # UPDATE STATE
                                # =============================

                                last_accepted_prediction = (

                                    stable_prediction

                                )


                                cooldown_counter = (

                                    COOLDOWN_FRAMES

                                )


                                prediction_history.clear()


            except Exception as e:

                print(

                    f"Prediction error: {e}"

                )

                current_prediction = (
                    "Unknown"
                )

                current_confidence = 0.0

                top_predictions = []


    # ========================================================
    # NO HAND DETECTED
    # ========================================================

    else:

        prediction_history.clear()

        last_accepted_prediction = None


    # ========================================================
    # COOLDOWN
    # ========================================================

    if cooldown_counter > 0:

        cooldown_counter -= 1


    # ========================================================
    # DRAW HAND LANDMARKS ON CAMERA
    # ========================================================

    if result.hand_landmarks:

        connections = (

            vision.HandLandmarksConnections.HAND_CONNECTIONS

        )


        for connection in connections:

            start_landmark = (

                hand_landmarks[

                    connection.start

                ]

            )


            end_landmark = (

                hand_landmarks[

                    connection.end

                ]

            )


            start_x = int(

                start_landmark.x

                *

                frame_width

            )


            start_y = int(

                start_landmark.y

                *

                frame_height

            )


            end_x = int(

                end_landmark.x

                *

                frame_width

            )


            end_y = int(

                end_landmark.y

                *

                frame_height

            )


            cv2.line(

                frame,

                (

                    start_x,

                    start_y

                ),

                (

                    end_x,

                    end_y

                ),

                WHITE,

                2,

                cv2.LINE_AA

            )


        for landmark in hand_landmarks:

            x = int(

                landmark.x

                *

                frame_width

            )


            y = int(

                landmark.y

                *

                frame_height

            )


            cv2.circle(

                frame,

                (

                    x,

                    y

                ),

                5,

                LANDMARK_GREEN,

                -1

            )


    # ========================================================
    # CREATE CANVAS
    # ========================================================

    canvas = np.full(

        (

            CANVAS_HEIGHT,

            CANVAS_WIDTH,

            3

        ),

        LIGHT_BG,

        dtype=np.uint8

    )


    # ========================================================
    # HEADER
    # ========================================================

    HEADER_HEIGHT = 85


    cv2.rectangle(

        canvas,

        (

            0,

            0

        ),

        (

            CANVAS_WIDTH,

            HEADER_HEIGHT

        ),

        DARK_NAVY,

        -1

    )


    # ========================================================
    # CAMERA STATUS
    # ========================================================

    status_color = (

        GREEN

        if hand_detected

        else RED

    )


    cv2.circle(

        canvas,

        (

            30,

            42

        ),

        8,

        status_color,

        -1

    )


    draw_text(

        canvas,

        "CAMERA",

        (

            45,

            48

        ),

        0.52,

        WHITE,

        2

    )


    # ========================================================
    # MAIN TITLE
    # ========================================================

    draw_text(

        canvas,

        "SIGN LANGUAGE RECOGNITION SYSTEM",

        (

            330,

            38

        ),

        0.95,

        WHITE,

        2

    )


    draw_text(

        canvas,

        "Real-Time Sign to Text | Autocomplete | Speech",

        (

            395,

            65

        ),

        0.48,

        WHITE,

        1

    )


    # ========================================================
    # DATE & TIME
    # ========================================================

    now = datetime.now()


    time_text = now.strftime(

        "%I:%M:%S %p"

    )


    date_text = now.strftime(

        "%d %b %Y"

    )


    draw_text(

        canvas,

        time_text,

        (

            1110,

            35

        ),

        0.50,

        WHITE,

        2

    )


    draw_text(

        canvas,

        date_text,

        (

            1125,

            62

        ),

        0.40,

        WHITE,

        1

    )


    # ========================================================
    # MAIN CONTENT
    # ========================================================

    CONTENT_TOP = 100


    # ========================================================
    # CAMERA PANEL
    # ========================================================

    camera_x = 15

    camera_y = CONTENT_TOP

    camera_w = 730

    camera_h = 410


    resized_camera = cv2.resize(

        frame,

        (

            camera_w,

            camera_h

        )

    )


    canvas[

        camera_y:
        camera_y + camera_h,

        camera_x:
        camera_x + camera_w

    ] = resized_camera


    cv2.rectangle(

        canvas,

        (

            camera_x,

            camera_y

        ),

        (

            camera_x + camera_w,

            camera_y + camera_h

        ),

        BLUE,

        3

    )


    # ========================================================
    # CAMERA STATUS BADGE
    # ========================================================

    badge_color = (

        GREEN

        if hand_detected

        else RED

    )


    badge_text = (

        "HAND DETECTED"

        if hand_detected

        else "NO HAND"

    )


    cv2.rectangle(

        canvas,

        (

            camera_x + 15,

            camera_y + 15

        ),

        (

            camera_x + 170,

            camera_y + 50

        ),

        badge_color,

        -1

    )


    draw_text(

        canvas,

        badge_text,

        (

            camera_x + 28,

            camera_y + 38

        ),

        0.42,

        WHITE,

        2

    )


    # ========================================================
    # LANDMARK INFO
    # ========================================================

    cv2.rectangle(

        canvas,

        (

            camera_x + 15,

            camera_y + camera_h - 40

        ),

        (

            camera_x + 150,

            camera_y + camera_h - 10

        ),

        BLACK,

        -1

    )


    draw_text(

        canvas,

        f"Landmarks: {21 if hand_detected else 0}",

        (

            camera_x + 25,

            camera_y + camera_h - 20

        ),

        0.40,

        WHITE,

        1

    )


    # ========================================================
    # FPS INFO
    # ========================================================

    cv2.rectangle(

        canvas,

        (

            camera_x + camera_w - 105,

            camera_y + camera_h - 40

        ),

        (

            camera_x + camera_w - 15,

            camera_y + camera_h - 10

        ),

        BLACK,

        -1

    )


    draw_text(

        canvas,

        f"FPS: {fps:.1f}",

        (

            camera_x + camera_w - 93,

            camera_y + camera_h - 20

        ),

        0.40,

        GREEN,

        1

    )


    # ========================================================
    # RIGHT SIDE PANELS
    # ========================================================

    right_x = 760

    right_w = 245

    top_x = 1020

    top_w = 245


    # ========================================================
    # CURRENT PREDICTION PANEL
    # ========================================================

    draw_panel(

        canvas,

        right_x,

        CONTENT_TOP,

        right_x + right_w,

        300,

        "CURRENT PREDICTION",

        NAVY

    )


    prediction_display = (

        current_prediction

        if hand_detected

        else

        "No Hand"

    )


    prediction_size = cv2.getTextSize(

        prediction_display,

        cv2.FONT_HERSHEY_SIMPLEX,

        1.05,

        2

    )[0]


    prediction_x = (

        right_x

        +

        (

            right_w

            -

            prediction_size[0]

        )

        //

        2

    )


    draw_text(

        canvas,

        prediction_display,

        (

            prediction_x,

            185

        ),

        1.05,

        GREEN,

        2

    )


    draw_text(

        canvas,

        "Confidence",

        (

            right_x + 15,

            225

        ),

        0.45,

        TEXT_DARK,

        1

    )


    draw_text(

        canvas,

        f"{current_confidence:.1f}%",

        (

            right_x + 160,

            225

        ),

        0.45,

        GREEN,

        2

    )


    draw_progress_bar(

        canvas,

        right_x + 15,

        240,

        right_x + right_w - 15,

        260,

        current_confidence

    )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    draw_panel(

        canvas,

        top_x,

        CONTENT_TOP,

        top_x + top_w,

        300,

        "TOP 3 PREDICTIONS",

        NAVY

    )


    top_y = 155


    for i in range(3):

        if i < len(
            top_predictions
        ):

            label, probability = (

                top_predictions[i]

            )

        else:

            label = "-"

            probability = 0.0


        cv2.rectangle(

            canvas,

            (

                top_x + 15,

                top_y - 20

            ),

            (

                top_x + 45,

                top_y + 10

            ),

            (

                235,

                235,

                235

            ),

            -1

        )


        draw_text(

            canvas,

            str(i + 1),

            (

                top_x + 25,

                top_y

            ),

            0.38,

            TEXT_DARK,

            2

        )


        draw_text(

            canvas,

            label,

            (

                top_x + 60,

                top_y

            ),

            0.48,

            GREEN if i == 0 else TEXT_DARK,

            2

        )


        draw_text(

            canvas,

            f"{probability:.1f}%",

            (

                top_x + 160,

                top_y

            ),

            0.40,

            GREEN if i == 0 else TEXT_DARK,

            1

        )


        top_y += 40


    # ========================================================
    # CURRENT WORD
    # ========================================================

    draw_panel(

        canvas,

        right_x,

        315,

        right_x + right_w,

        425,

        "CURRENT WORD",

        NAVY

    )


    word_display = (

        current_word

        if current_word

        else

        "-"

    )


    word_size = cv2.getTextSize(

        word_display,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.90,

        2

    )[0]


    word_x = (

        right_x

        +

        (

            right_w

            -

            word_size[0]

        )

        //

        2

    )


    draw_text(

        canvas,

        word_display,

        (

            word_x,

            390

        ),

        0.90,

        BLUE,

        2

    )


    # ========================================================
    # WORD SUGGESTIONS
    # ========================================================

    draw_panel(

        canvas,

        top_x,

        315,

        top_x + top_w,

        425,

        "WORD SUGGESTIONS",

        DARK_GREEN

    )


    if suggestions:

        y = 365


        for i, suggestion in enumerate(

            suggestions[:3],

            start=1

        ):

            draw_text(

                canvas,

                f"{i}.",

                (

                    top_x + 20,

                    y

                ),

                0.45,

                DARK_GREEN,

                2

            )


            draw_text(

                canvas,

                suggestion,

                (

                    top_x + 50,

                    y

                ),

                0.45,

                TEXT_DARK,

                1

            )


            y += 25

    else:

        draw_text(

            canvas,

            "No suggestions",

            (

                top_x + 20,

                375

            ),

            0.42,

            TEXT_GRAY,

            1

        )


    # ========================================================
    # SENTENCE PANEL
    # ========================================================

    sentence_x = 15

    sentence_y = 525

    sentence_w = 730

    sentence_h = 105


    draw_panel(

        canvas,

        sentence_x,

        sentence_y,

        sentence_x + sentence_w,

        sentence_y + sentence_h,

        "RECOGNIZED SENTENCE",

        BLUE

    )


    display_text = (
        sentence.strip()
    )


    if current_word:

        if display_text:

            display_text += " "

        display_text += current_word


    if not display_text:

        display_text = (
            "Start signing..."
        )


    # Keep latest visible text
    display_text = (
        display_text[-50:]
    )


    draw_text(

        canvas,

        display_text,

        (

            sentence_x + 20,

            sentence_y + 72

        ),

        0.72,

        TEXT_DARK,

        2

    )


    # ========================================================
    # SPEAK BUTTON INSIDE SENTENCE
    # ========================================================

    speaker_x1 = (

        sentence_x

        +

        sentence_w

        -

        70

    )


    speaker_y1 = (

        sentence_y

        +

        48

    )


    speaker_x2 = (

        sentence_x

        +

        sentence_w

        -

        15

    )


    speaker_y2 = (

        sentence_y

        +

        92

    )


    cv2.rectangle(

        canvas,

        (

            speaker_x1,

            speaker_y1

        ),

        (

            speaker_x2,

            speaker_y2

        ),

        ORANGE,

        -1

    )


    draw_text(

        canvas,

        "T",

        (

            speaker_x1 + 19,

            speaker_y1 + 30

        ),

        0.55,

        WHITE,

        2

    )


    # ========================================================
    # SPACE PANEL
    # ========================================================

    draw_panel(

        canvas,

        right_x,

        440,

        right_x + right_w,

        515,

        "SPACE",

        PURPLE

    )


    draw_text(

        canvas,

        "Complete current word",

        (

            right_x + 12,

            475

        ),

        0.40,

        TEXT_DARK,

        1

    )


    draw_text(

        canvas,

        "Gesture / SPACE key",

        (

            right_x + 12,

            500

        ),

        0.36,

        PURPLE,

        1

    )


    # ========================================================
    # DELETE PANEL
    # ========================================================

    draw_panel(

        canvas,

        top_x,

        440,

        top_x + top_w,

        515,

        "DELETE",

        RED

    )


    draw_text(

        canvas,

        "Delete character / word",

        (

            top_x + 12,

            475

        ),

        0.40,

        TEXT_DARK,

        1

    )


    draw_text(

        canvas,

        "Gesture / BACKSPACE",

        (

            top_x + 12,

            500

        ),

        0.36,

        RED,

        1

    )


    # ========================================================
    # BOTTOM CONTROL BUTTONS
    #
    # IMPORTANT:
    # These buttons are now at Y=650.
    # They are fully inside 800px canvas.
    # ========================================================

    button_y1 = 650

    button_y2 = 710

    button_w = 190

    button_gap = 14


    buttons = [

        (

            "C",

            "CLEAR",

            "Clear sentence",

            BLUE

        ),

        (

            "T",

            "SPEAK",

            "Speak text",

            ORANGE

        ),

        (

            "1",

            "ACCEPT 1",

            "Suggestion 1",

            GREEN

        ),

        (

            "2",

            "ACCEPT 2",

            "Suggestion 2",

            YELLOW

        ),

        (

            "3",

            "ACCEPT 3",

            "Suggestion 3",

            PURPLE

        ),

        (

            "Q",

            "QUIT",

            "Exit system",

            RED

        )

    ]


    for i, (

        key_name,

        title,

        description,

        color

    ) in enumerate(buttons):

        x1 = (

            15

            +

            i

            *

            (

                button_w

                +

                button_gap

            )

        )


        x2 = (

            x1

            +

            button_w

        )


        draw_button(

            canvas,

            x1,

            button_y1,

            x2,

            button_y2,

            key_name,

            title,

            description,

            color

        )


    # ========================================================
    # FOOTER
    # ========================================================

    FOOTER_Y = 735


    cv2.rectangle(

        canvas,

        (

            0,

            FOOTER_Y

        ),

        (

            CANVAS_WIDTH,

            CANVAS_HEIGHT

        ),

        WHITE,

        -1

    )


    draw_text(

        canvas,

        "Model: Random Forest",

        (

            15,

            760

        ),

        0.38,

        TEXT_GRAY,

        1

    )


    draw_text(

        canvas,

        f"| Features: 63 | Classes: {len(class_labels)}",

        (

            170,

            760

        ),

        0.38,

        TEXT_GRAY,

        1

    )


    draw_text(

        canvas,

        "SYSTEM READY",

        (

            570,

            760

        ),

        0.40,

        DARK_GREEN,

        2

    )


    draw_text(

        canvas,

        "SPACE | DEL | 1 | 2 | 3 | C | T | Q",

        (

            900,

            760

        ),

        0.38,

        TEXT_GRAY,

        1

    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(

        WINDOW_NAME,

        canvas

    )


    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    key = (

        cv2.waitKey(1)

        &

        0xFF

    )


    # ========================================================
    # QUIT
    # ========================================================

    if key == ord(
        "q"
    ):

        break


    # ========================================================
    # CLEAR
    # ========================================================

    elif key == ord(
        "c"
    ):

        sentence = ""

        current_word = ""

        suggestions = []

        prediction_history.clear()

        last_accepted_prediction = None

        cooldown_counter = 0


    # ========================================================
    # SPEAK
    # ========================================================

    elif key == ord(
        "t"
    ):

        speak_sentence()


    # ========================================================
    # SPACE
    # ========================================================

    elif key == 32:

        if current_word:

            if (

                sentence

                and

                not sentence.endswith(
                    " "
                )

            ):

                sentence += " "


            sentence += (
                current_word
            )


            sentence += " "


            current_word = ""

            suggestions = []


        elif (

            sentence

            and

            not sentence.endswith(
                " "
            )

        ):

            sentence += " "


    # ========================================================
    # DELETE / BACKSPACE
    # ========================================================

    elif key in [

        8,

        127

    ]:

        delete_character()


    # ========================================================
    # ACCEPT SUGGESTION 1
    # ========================================================

    elif key == ord(
        "1"
    ):

        if len(
            suggestions
        ) >= 1:

            accept_suggestion(

                suggestions[0]

            )


    # ========================================================
    # ACCEPT SUGGESTION 2
    # ========================================================

    elif key == ord(
        "2"
    ):

        if len(
            suggestions
        ) >= 2:

            accept_suggestion(

                suggestions[1]

            )


    # ========================================================
    # ACCEPT SUGGESTION 3
    # ========================================================

    elif key == ord(
        "3"
    ):

        if len(
            suggestions
        ) >= 3:

            accept_suggestion(

                suggestions[2]

            )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

detector.close()

cv2.destroyAllWindows()


# ============================================================
# FINAL SENTENCE
# ============================================================

final_text = (
    sentence.strip()
)


if current_word:

    if final_text:

        final_text += " "

    final_text += (
        current_word
    )


print()

print(
    "=" * 70
)

print(
    "FINAL SENTENCE:"
)

print(
    final_text
)

print(
    "=" * 70
)