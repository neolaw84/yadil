from typing import Union, Callable, List

import pandas as pd
import fire

from yadil.image.face import extract_all


class ImageUtils(object):
    def extract_all(self, input_glob, output_dir, input_meta, output_meta):
        extract_all(input_glob, output_dir, input_meta, output_meta)

if __name__ == "__main__":
    fire.Fire(ImageUtils())