from datetime import datetime
import re
from typing import List, Set
from enum import Enum

from cv2 import Mat
import cv2
import pytesseract
from fuzzywuzzy import fuzz

from pipeline import Pipeline

ValidationStatus = Enum("ValidationStatus", "COMPLETE PARTIAL_YEAR PARTIAL_NAME FAILED")
ImageOrientation = Enum("ImageOrientation", "NORMAL CLOCKWISE COUNTER_CLOCKWISE")


class Validator:

    def __init__(self, pipelines: List[Pipeline], img_path: str, file_base_name: str, name: str, dob: datetime):
        self._initialize_date_things()
        self._initialize_name_things(name)

        try:
            self.img = cv2.imread(img_path)
        except Exception as e:
            raise Exception("Error reading image:", e)

        self.img_rotated_clockwise = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        self.img_rotated_counter_clockwise = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.file_base_name = file_base_name
        self.name = name
        self.dob = dob
        self.pipelines = pipelines
        self.validation_status = ValidationStatus.FAILED

        self.succeeded = {}
        self.failed: Set[str] = set()
        pass

    def _initialize_date_things(self):

        # TODO: Add support for string months (Jan, Feb, etc)

        # Regex to find a date in the format of yyyy-mm-dd or mm-dd-yyyy with any separator (space, dash, or slash)
        full_year_numerical_date = r'\b\d{8}\b'
        last_two_year_digits_numerical_date = r'\b\d{6}\b'
        date_re = re.compile("(%s|%s)" % (full_year_numerical_date, last_two_year_digits_numerical_date))
        self.date_re = date_re

        # Used to check if the regex matches an expected date pattern
        valid_date_patterns = []
        valid_date_patterns.append(f'%Y%m%d')
        valid_date_patterns.append(f'%y%m%d')

        valid_date_patterns.append(f'%d%m%Y')
        valid_date_patterns.append(f'%d%m%y')

        valid_date_patterns.append(f'%m%d%Y')
        valid_date_patterns.append(f'%m%d%y')
        self.valid_date_patterns = valid_date_patterns

        # Used to check if we found a valid year or full date of birth
        self.is_valid_year: bool = False
        self.is_valid_full_dob: bool = False

    def _initialize_name_things(self, name: str):
        # split the name into parts and remove any strings that are less than 1 character like the "D." John D. Smith
        self.all_names = [n.strip() for n in name.split(
            " ") if len(n.strip().strip(".")) > 1]

        if len(self.all_names) < 2:
            raise ValueError("Name must have at least two parts")

        # Used to check how many of the names we found
        self.found_names = set()

        # Used to check if we found a valid name
        self.is_valid_name = False

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
            self.is_valid_name = True

    def _check_dob(self, data: str):
        # Restrict to just numbers
        numerical_data = re.sub("[^0-9\n]", "", data)

        # Sometimes we'll read 1979 as 4979 where the year can be the first attribute or the last attribute in the DOB
        # This is a hack to fix that
        if str(self.dob.year)[0] == "1":
            fuzzy_year = "4" + str(self.dob.year)[1:]
            if fuzzy_year in numerical_data:
                numerical_data = numerical_data.replace(
                    fuzzy_year, str(self.dob.year))
                self.is_valid_year = True

        matched_dates = []
        matches = self.date_re.findall(numerical_data)

        for date_str in matches:
            for pattern in self.valid_date_patterns:
                try:
                    matched_dates.append(datetime.strptime(date_str, pattern))
                except ValueError:
                    pass

        for matched_date in matched_dates:
            if matched_date.year == self.dob.year:
                self.is_valid_year = True
            if matched_date.year == self.dob.year and matched_date.month == self.dob.month and matched_date.day == self.dob.day:
                self.is_valid_full_dob = True

        if str(self.dob.year) in numerical_data:
            self.is_valid_year = True

    def validate(self, orientation: ImageOrientation = ImageOrientation.NORMAL) -> None:

        print(f"Validating {self.file_base_name} with {orientation} orientation...")

        for i, pipeline in enumerate(self.pipelines):
            if orientation == ImageOrientation.NORMAL:
                processed_img = pipeline.execute(self.img)
            elif orientation == ImageOrientation.CLOCKWISE:
                processed_img = pipeline.execute(self.img_rotated_clockwise)
            elif orientation == ImageOrientation.COUNTER_CLOCKWISE:
                processed_img = pipeline.execute(self.img_rotated_counter_clockwise)
            else:
                raise ValueError("Invalid orientation")

            data = pytesseract.image_to_string(processed_img).lower()

            self._check_name(data)

            self._check_dob(data)

            # print(data)

            # print(self.found_names, self.is_valid_year,
            #       self.is_valid_full_dob, self.is_valid_name)
            # print("#" * 100)

            if self.is_valid_full_dob and self.is_valid_name:
                self.validation_status = ValidationStatus.COMPLETE
            elif self.is_valid_year and self.is_valid_name:
                self.validation_status = ValidationStatus.PARTIAL_YEAR
            elif self.is_valid_full_dob and len(self.found_names) >= 1:
                self.validation_status = ValidationStatus.PARTIAL_NAME

            if self.validation_status != ValidationStatus.FAILED:
                print(f"{self.file_base_name} completed ({self.validation_status}) in {i} pipelines")
                return

        print(f"Failed to validate {self.file_base_name} with {orientation}")
        return

    def get_status(self) -> ValidationStatus:
        return self.validation_status

    def get_self(self):
        return self


def validate_async(validator: Validator):
    try:
        validator.validate(ImageOrientation.NORMAL)
        if validator.get_status() == ValidationStatus.FAILED:
            validator.validate(ImageOrientation.CLOCKWISE)
        if validator.get_status() == ValidationStatus.FAILED:
            validator.validate(ImageOrientation.COUNTER_CLOCKWISE)

    except Exception as e:
        print("Error validating", validator.file_base_name, e)

    return validator.get_self()
