from datetime import datetime
import re
from typing import List, Set
from cv2 import Mat
import pytesseract

from pipeline import Pipeline


class Validator:

    def __init__(self, pipelines: List[Pipeline], img: Mat, file_base_name: str, name: str, dob: datetime):
        self._initialize_date_things()
        self._initialize_name_things(name)

        self.img = img
        self.file_base_name = file_base_name
        self.name = name
        self.dob = dob
        self.pipelines = pipelines

        self.succeeded = {}
        self.failed: Set[str] = set()
        pass

    def _initialize_date_things(self):
        # Regex to find a date in the format of yyyy-mm-dd or mm-dd-yyyy with any separator (space, dash, or slash)
        yyyy_mm_dd = r'\b\d{4}[ /-]\d{2}[ /-]\d{2}\b'
        mm_dd_yyyy = r'\b\d{2}[ /-]\d{2}[ /-]\d{4}\b'
        date_re = re.compile("(%s|%s)" % (yyyy_mm_dd, mm_dd_yyyy))
        self.date_re = date_re

        # Used to check if the regex matches an expected date pattern
        valid_date_patterns = []
        separators = ['-', '/', ' ']
        for sep in separators:
            valid_date_patterns.append(f'%Y{sep}%m{sep}%d')
            valid_date_patterns.append(f'%y{sep}%m{sep}%d')

            valid_date_patterns.append(f'%d{sep}%m{sep}%Y')
            valid_date_patterns.append(f'%d{sep}%m{sep}%y')

            valid_date_patterns.append(f'%m{sep}%d{sep}%Y')
            valid_date_patterns.append(f'%m{sep}%d{sep}%y')
        self.valid_date_patterns = valid_date_patterns

        # Used to check if we found a valid year or full date of birth
        self.is_valid_year: bool = False
        self.is_valid_full_dob: bool = False

    def _initialize_name_things(self, name: str):
        self.all_names = [n.strip() for n in name.split(" ") if len(n.strip().strip(".")) > 1]

        if len(self.all_names) < 2:
            raise ValueError("Name must have at least two parts")

        # Used to check how many of the names we found
        self.found_names = set()

        # Used to check if we found a valid name
        self.is_valid_name = False

    def _check_name(self, data: str):
        for name in self.all_names:
            if name in data:
                self.found_names.add(name)

        # If we found at least two names, we have a valid name
        if len(self.found_names) >= 2:
            self.is_valid_name = True
    
    def _check_dob(self, data: str):
        matched_dates = []
        matches = self.date_re.findall(data)
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

        if str(self.dob.year) in data:
            self.is_valid_year = True

    def validate(self):

        print(f"Validating {self.file_base_name}...")

        for pipeline in self.pipelines:
            processed_img = pipeline.execute(self.img)

            data = pytesseract.image_to_string(processed_img).lower()

            self._check_name(data)

            self._check_dob(data)

            if self.is_valid_full_dob and self.is_valid_name:
                print(f"Success complete", self.file_base_name, pipeline)
                return
            elif self.is_valid_year and self.is_valid_name:
                print(f"Success with partial Year", self.file_base_name, pipeline)
                return
            elif self.is_valid_full_dob and len(self.found_names) >= 1:
                print(f"Success with partial name", self.file_base_name, pipeline)
                return
        
        print(f"Failed to validate", self.file_base_name)