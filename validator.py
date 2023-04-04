from datetime import datetime
import re
from typing import List
from enum import Enum

from cv2 import Mat
import cv2
import pytesseract
from fuzzywuzzy import fuzz
import face_recognition
from pdf2image.pdf2image import convert_from_path
from pipeline import Pipeline
import numpy as np

ImageOrientation = Enum(
    "ImageOrientation", "NORMAL CLOCKWISE COUNTER_CLOCKWISE")

ValidationStates = Enum(
    "ValidationStates", "COMPLETE PARTIAL FAILED")


class ValidationStatus:
    def __init__(self):
        self.status = ValidationStates.FAILED

    def __str__(self) -> str:
        return str(self.status.name)

    def is_complete(self) -> bool:
        return self.status == ValidationStates.COMPLETE

    def is_partial(self) -> bool:
        return self.status == ValidationStates.PARTIAL

    def is_failed(self) -> bool:
        return self.status == ValidationStates.FAILED

    def update(self, status: ValidationStates):
        if self.is_complete():
            return
        self.status = status


class Validator:

    # This is common in passports
    string_month_map = {
        "janjan": "01",
        "febfev": "02",
        "febfév": "02",
        "marmar": "03",
        "apravr": "04",
        "maymai": "05",
        "junjui": "06",
        "junejuin": "06",
        "juljul": "07",
        "julyjuil": "07",
        "augaoû": "08",
        "augaou": "08",
        "sepsep": "09",
        "septsept": "09",
        "sepsept": "09",  # british passport has this instead of sep
        "octoct": "10",
        "novnov": "11",
        "decdec": "12",
        "decdéc": "12",
    }

    def __init__(self, pipelines: List[Pipeline], base_name: str, id_path: str, headshot_path: str, name: str, dob: datetime):
        try:
            if id_path.endswith(".pdf"):
                self.id = np.array(convert_from_path(id_path)[0])
            else:
                self.id = cv2.imread(id_path)
        except Exception as e:
            raise Exception("Error reading id:", e)
        self.id_rotated_clockwise = cv2.rotate(
            self.id, cv2.ROTATE_90_CLOCKWISE)
        self.id_rotated_counter_clockwise = cv2.rotate(
            self.id, cv2.ROTATE_90_COUNTERCLOCKWISE)

        try:
            if headshot_path.endswith(".pdf"):
                self.headshot = np.array(convert_from_path(headshot_path)[0])
            else:
                self.headshot = cv2.imread(headshot_path)
        except Exception as e:
            raise Exception("Error reading headshot:", e)

        self.base_name = base_name
        self.name = name
        self.dob = dob
        self.pipelines = pipelines

        self._initialize_headshot_things()
        self._initialize_date_things()
        self._initialize_name_things()

    def _initialize_headshot_things(self):
        self.headshot_status = ValidationStatus()

        try:
            self.headshot_encoding = face_recognition.face_encodings(
                self.headshot, model="large")[0]
        except Exception:
            self.headshot_encoding = None
            return

    def _initialize_date_things(self):
        self.dob_status = ValidationStatus()

        # Regex to find different date formats
        date_formats = [
            r"\d{8}",  # eg. 19760508
            r"\d{6}",  # eg. 080576
            r"\d{2}[a-zA-Z]{3}\d{4}",  # eg. 08May1976
            r"\d{4}[a-zA-Z]{3}\d{2}",  # eg. 1976May08
            r"\d{2}[a-zA-Z]{3}\d{2}",  # eg. 08May76 or 76May08
            r"[a-zA-Z]{3}\d{2}\d{4}",  # eg. May081976
            r"[a-zA-Z]{3}\d{2}\d{2}",  # eg. May0876
        ]

        date_re = re.compile(f"({'|'.join(date_formats)})")

        self.date_re = date_re

        # Used to check if the regex matches an expected date pattern
        valid_date_patterns = []
        valid_date_patterns.append(f'%Y%m%d')  # eg. 19760508
        valid_date_patterns.append(f'%y%m%d')  # eg. 760508

        valid_date_patterns.append(f'%d%m%Y')  # eg. 08051976
        valid_date_patterns.append(f'%d%m%y')  # eg. 080576

        valid_date_patterns.append(f'%m%d%Y')  # eg. 05081976
        valid_date_patterns.append(f'%m%d%y')  # eg. 050876

        valid_date_patterns.append(f'%Y%b%d')  # eg. 1976May08
        valid_date_patterns.append(f'%y%b%d')  # eg. 76May08

        valid_date_patterns.append(f'%d%b%Y')  # eg. 08May1976
        valid_date_patterns.append(f'%d%b%y')  # eg. 08May76

        valid_date_patterns.append(f'%b%d%Y')  # eg. May081976
        valid_date_patterns.append(f'%b%d%y')  # eg. May0876

        self.valid_date_patterns = valid_date_patterns

    def _initialize_name_things(self):
        self.name_status = ValidationStatus()

        # split the name into parts and remove any strings that are less than 1 character like the "D." John D. Smith
        self.all_names = [n.strip() for n in self.name.split(
            " ") if len(n.strip().strip(".")) > 1]

        if len(self.all_names) < 2:
            raise ValueError("Name must have at least two parts")

        # Used to check how many of the names we found
        self.found_names = set()

    def _check_headshot(self, id_image: Mat):
        if self.headshot_encoding is None:
            return

        try:
            id_encoding = face_recognition.face_encodings(
                id_image, model="large")[0]
        except Exception:
            return

        # Lower tolerance means more strict
        if face_recognition.compare_faces([self.headshot_encoding], id_encoding, tolerance=0.6)[0]:
            self.headshot_status.update(ValidationStates.COMPLETE)

    def _check_name(self, data: str):

        # Remove all non-alphabetical characters
        words = re.sub(r'[^a-zA-Z\s]', '', data).split()

        for name in self.all_names:
            for word in words:
                score = fuzz.token_set_ratio(name, word)
                if score > 85:
                    self.found_names.add(name)

        # If we found at least two names, we have a valid name
        if len(self.found_names) >= 2:
            self.name_status.update(ValidationStates.COMPLETE)
        elif len(self.found_names) >= 1:
            self.name_status.update(ValidationStates.PARTIAL)

    def _check_dob(self, data: str):
        # Restrict to just numbers
        cleaned_data = re.sub("[^0-9a-z\n]", "", data)

        for string_month in self.string_month_map:
            if string_month in cleaned_data:
                # We need to replace the string month with the number month
                # but incase we have multiple matches ie. sepsep and sepsept, we need to make a copy so we get every combination
                cleaned_data += "\nNEW STUFF\n" + cleaned_data.replace(
                    string_month, self.string_month_map[string_month])

        # Sometimes we'll read 1979 as 4979 where the year can be the first attribute or the last attribute in the DOB
        # This is a hack to fix that
        if str(self.dob.year)[0] == "1":
            fuzzy_year = "4" + str(self.dob.year)[1:]  # eg. 1979 -> 4979
            if fuzzy_year in cleaned_data:
                cleaned_data = cleaned_data.replace(
                    fuzzy_year, str(self.dob.year))

        matched_dates = []
        matches = self.date_re.findall(cleaned_data)

        for date_str in matches:
            for pattern in self.valid_date_patterns:
                try:
                    curr_dt = datetime.strptime(date_str, pattern)
                    if "%y" in pattern:
                        # sometimes 2 number years ie. 58 for 1958 will be read as 2058
                        if curr_dt.year > 2023:
                            matched_dates.append(
                                curr_dt.replace(year=curr_dt.year-100))
                    matched_dates.append(curr_dt)
                except ValueError:
                    pass

        if str(self.dob.year) in cleaned_data:
            self.dob_status.update(ValidationStates.PARTIAL)

        for matched_date in matched_dates:
            if matched_date.year == self.dob.year:
                self.dob_status.update(ValidationStates.PARTIAL)
            if matched_date.year == self.dob.year and matched_date.month == self.dob.month and matched_date.day == self.dob.day:
                self.dob_status.update(ValidationStates.COMPLETE)

    def validate(self, orientation: ImageOrientation = ImageOrientation.NORMAL) -> None:

        # print(
        #     f"Validating {self.base_name} with {orientation} orientation...")

        if orientation == ImageOrientation.NORMAL:
            current_id = self.id
        elif orientation == ImageOrientation.CLOCKWISE:
            current_id = self.id_rotated_clockwise
        elif orientation == ImageOrientation.COUNTER_CLOCKWISE:
            current_id = self.id_rotated_counter_clockwise
        else:
            raise ValueError("Invalid orientation")

        if not self.headshot_status.is_complete():
            self._check_headshot(current_id)

        # Skip if we already have validation for name and dob
        if self.is_valid_id():
            return

        for pipeline in self.pipelines:
            processed_img = pipeline.execute(current_id)

            data = pytesseract.image_to_string(processed_img).lower()

            if not self.name_status.is_complete():
                self._check_name(data)

            if not self.dob_status.is_complete():
                self._check_dob(data)

            # print(data)

            # print(self.found_names, self.validation_status_string())
            # print("#" * 100)

            if self.is_valid_id():
                return
        return

    def is_valid_id(self) -> bool:
        if self.dob_status.is_complete() and self.name_status.is_complete():
            return True
        elif self.dob_status.is_partial() and self.name_status.is_complete():
            return True
        elif self.dob_status.is_complete() and self.name_status.is_partial():
            return True
        return False

    def is_valid(self) -> bool:
        return self.headshot_status.is_complete() and self.is_valid_id()

    def validation_status_string(self) -> str:
        return f"HEADSHOT: {self.headshot_status} | DOB: {self.dob_status} | NAME: {self.name_status}"


def validate_async(validator: Validator) -> Validator:

    print(
        f"Validating {validator.base_name}...")

    try:
        validator.validate(ImageOrientation.NORMAL)
        # If we don't have a valid ID, try rotating the image
        if not validator.is_valid():
            validator.validate(ImageOrientation.CLOCKWISE)
        if not validator.is_valid():
            validator.validate(ImageOrientation.COUNTER_CLOCKWISE)

        if not validator.is_valid():
            print(
                f"Failed to validate {validator.base_name}: {validator.validation_status_string()}")
        else:
            print(
                f"Successfully validated {validator.base_name}: {validator.validation_status_string()}")

    except Exception as e:
        print("Error validating", validator.base_name, e)

    return validator
