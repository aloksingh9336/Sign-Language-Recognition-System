import os
import csv
import time
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "ASL_Alphabet_Dataset"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "landmark_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)


# Maximum number of images processed per class.
# Start with 300 for fast testing.
# After everything works, increase to 500 or 1000.
MAX_IMAGES_PER_CLASS = 500


# Classes we want to process
CLASSES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z",
    "del",
    "nothing",
    "space"
]


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

def create_hand_landmarker():

    print("=" * 70)
    print("INITIALIZING MEDIAPIPE HAND LANDMARKER")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):

        print()
        print("ERROR: MediaPipe model not found!")
        print()
        print("Expected model:")
        print(MODEL_PATH)
        print()

        return None

    try:

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        detector = vision.HandLandmarker.create_from_options(
            options
        )

        print()
        print("MediaPipe initialized successfully.")
        print()

        return detector

    except Exception as e:

        print()
        print("ERROR INITIALIZING MEDIAPIPE")
        print(str(e))
        print()

        return None


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_landmarks(detector, image_path):

    try:

        image = cv2.imread(image_path)

        if image is None:
            return None

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Convert NumPy array to MediaPipe Image.
        # This works with MediaPipe 0.10.35.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        # Detect hand
        result = detector.detect(mp_image)

        # No hand found
        if not result.hand_landmarks:
            return None

        # First detected hand
        landmarks = result.hand_landmarks[0]

        if len(landmarks) != 21:
            return None

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        # Use wrist as origin.
        wrist = landmarks[0]

        features = []

        for landmark in landmarks:

            x = landmark.x - wrist.x
            y = landmark.y - wrist.y
            z = landmark.z - wrist.z

            features.extend([
                x,
                y,
                z
            ])

        return features

    except Exception:

        return None


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_files(folder):

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    files = []

    if not os.path.exists(folder):
        return files

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            valid_extensions
        ):

            files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    # Sort for consistent processing
    files.sort()

    # Limit number of images
    files = files[
        :MAX_IMAGES_PER_CLASS
    ]

    return files


# ============================================================
# MAIN DATASET PROCESSING
# ============================================================

def main():

    start_time = time.time()

    print()
    print("=" * 70)
    print("SIGN LANGUAGE DATASET PREPARATION")
    print("=" * 70)

    print()
    print("Dataset directory:")
    print(DATASET_DIR)

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    print()
    print("Maximum images per class:")
    print(MAX_IMAGES_PER_CLASS)

    print()

    # Check dataset directory
    if not os.path.exists(DATASET_DIR):

        print("ERROR: Dataset directory does not exist.")

        return

    # Create MediaPipe detector
    detector = create_hand_landmarker()

    if detector is None:

        print()
        print("Cannot continue without MediaPipe.")
        return

    # Create output directory
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # --------------------------------------------------------
    # CSV HEADER
    # --------------------------------------------------------

    header = ["label"]

    for i in range(21):

        header.append(
            f"x{i}"
        )

        header.append(
            f"y{i}"
        )

        header.append(
            f"z{i}"
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_images = 0
    successful_images = 0
    failed_images = 0

    class_statistics = {}

    # --------------------------------------------------------
    # OPEN CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            header
        )

        # ====================================================
        # PROCESS EACH CLASS
        # ====================================================

        for class_name in CLASSES:

            print()
            print("-" * 70)
            print(
                f"Processing class: {class_name}"
            )
            print("-" * 70)

            class_folder = os.path.join(
                DATASET_DIR,
                class_name
            )

            image_files = get_image_files(
                class_folder
            )

            total_class_images = len(
                image_files
            )

            successful_class_images = 0

            print(
                f"Found {total_class_images} images."
            )

            if total_class_images == 0:

                print(
                    f"WARNING: No images found for {class_name}"
                )

                class_statistics[
                    class_name
                ] = (
                    0,
                    0
                )

                continue

            # ----------------------------------------------
            # PROCESS IMAGES
            # ----------------------------------------------

            for index, image_path in enumerate(
                image_files,
                start=1
            ):

                total_images += 1

                features = extract_landmarks(
                    detector,
                    image_path
                )

                if features is not None:

                    row = [
                        class_name
                    ]

                    row.extend(
                        features
                    )

                    writer.writerow(
                        row
                    )

                    successful_images += 1

                    successful_class_images += 1

                else:

                    failed_images += 1

                # Progress display
                if (
                    index % 25 == 0
                    or index == total_class_images
                ):

                    print(
                        f"\rProgress: "
                        f"{index}/{total_class_images} "
                        f"| Valid: "
                        f"{successful_class_images}",
                        end=""
                    )

            print()

            print(
                f"Completed {class_name}: "
                f"{successful_class_images} valid / "
                f"{total_class_images} total"
            )

            class_statistics[
                class_name
            ] = (
                successful_class_images,
                total_class_images
            )

    # ========================================================
    # CLOSE MEDIAPIPE
    # ========================================================

    detector.close()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    elapsed_time = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print("DATASET PROCESSING COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Total images processed: "
        f"{total_images}"
    )

    print(
        f"Successfully processed: "
        f"{successful_images}"
    )

    print(
        f"Failed / no hand detected: "
        f"{failed_images}"
    )

    print(
        f"Processing time: "
        f"{elapsed_time:.2f} seconds"
    )

    print()
    print("Output CSV:")
    print(OUTPUT_FILE)

    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CLASS-WISE SUMMARY")
    print("=" * 70)

    for class_name in CLASSES:

        valid_count, total_count = (
            class_statistics.get(
                class_name,
                (0, 0)
            )
        )

        print(
            f"{class_name:10s}: "
            f"{valid_count} valid / "
            f"{total_count} total"
        )

    print()
    print("=" * 70)

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if successful_images == 0:

        print()
        print(
            "ERROR: No hand landmarks were extracted."
        )

        print()
        print(
            "Possible reasons:"
        )

        print(
            "1. MediaPipe model is incorrect."
        )

        print(
            "2. Dataset images do not contain visible hands."
        )

        print(
            "3. Dataset path is incorrect."
        )

        print(
            "4. MediaPipe failed to detect hands."
        )

    else:

        print()
        print(
            "SUCCESS!"
        )

        print(
            f"{successful_images} landmark samples "
            f"are ready for model training."
        )

        print()
        print(
            "Next step:"
        )

        print(
            "Run train_model.py"
        )

    print()
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()