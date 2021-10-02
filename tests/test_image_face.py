import pathlib
from unittest import TestCase
import os
import glob

import numpy as np
import cv2

from tests import pytest_happy

from yadil.image.face import extract, extract_pitch_yaw_roll

THIS_FILE_PATH = pathlib.Path(__file__)
THIS_DIR_PATH = THIS_FILE_PATH.parent
IMAGE_DIR_PATH = pathlib.Path.joinpath(THIS_DIR_PATH, "images-with-faces")


class TestUtils(TestCase):
    def setUp(self) -> None:
        IMG_1_PATH = pathlib.Path.joinpath(IMAGE_DIR_PATH, "pexels-photo-3866555.png")
        IMG_2_PATH = pathlib.Path.joinpath(IMAGE_DIR_PATH, "photo-1601412436009-d964bd02edbc.png")
        self.images = [IMG_1_PATH, IMG_2_PATH]
        return super().setUp()

    def tearDown(self) -> None:
        for f in glob.glob(str(IMAGE_DIR_PATH) + "/result*.jpg"):
            os.remove(f)
        return super().tearDown()

    def test_extract_box(self):
        for ip in self.images:
            img = cv2.imread(str(ip))
            rimgs = extract(img)
            for i, rimg in enumerate(rimgs):
                self.assertTupleEqual(rimg.shape, (256, 256, 3))
                cv2.imwrite(str(IMAGE_DIR_PATH) + "/result-" + str(ip.name) + str(i) + ".jpg", rimg)

    def test_extract_rotate(self):
        for ip in self.images:
            img = cv2.imread(str(ip))
            rimgs = extract(img, correct_rotate=True)
            for i, rimg in enumerate(rimgs):
                self.assertTupleEqual(rimg.shape, (256, 256, 3))
                cv2.imwrite(str(IMAGE_DIR_PATH) + "/result-rotate-" + str(ip.name) + str(i) + ".jpg", rimg)

    def test_extract_rotate_vector(self):
        for ip in self.images:
            img = cv2.imread(str(ip))
            rimgs = extract_pitch_yaw_roll(img)
            for result in rimgs:
                assert isinstance(result, np.ndarray)
                assert len(result.shape) == 1
                assert result.shape[0] == 3
