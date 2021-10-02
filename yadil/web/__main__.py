from typing import Union, Callable, List

import pandas as pd
import fire

from yadil.web.scraper_config import default_config as scraper_config
from yadil.web.scraper import main as scraper_main
from yadil.web.utils import merge_meta_files, create_meta_file_from_glob


class WebUtils(object):
    def scrape(
        self,
        url_prefix: str = "http://localhost:8080/page-{page_id}",
        pg_start: int = 1,
        pg_end: int = 2,
        image_types: List = ["image/jpeg", "image/png", "image/jpg"],
        max_level=3,
        output_dir="~/outputs/",
        meta_file="meta.csv",
        get_urls: Union[str, Callable] = scraper_config.get_urls,
    ):
        """download image files from a series of web-pages into output_dir.

        Note: press 'q' to quit from `less`-like or `vim`-like environment.

        Args:
            url_prefix (str, optional): Prefix of the urls to get series of web-pages. You need to include `{page-id}` 
            placeholder to have page from `pg_start` to `pg_end` to progress. Defaults to "http://localhost:8080/page-{page_id}".
            pg_start (int, optional): Starting page to be placed in `{page-id}` placeholder. Defaults to 1.
            pg_end (int, optional): Ending page to be placed in `{page-id}` placeholder. Defaults to 2.
            image_types (List, optional): A list of meme-types to download. Defaults to ["image/jpeg", "image/png", "image/jpg"].
            max_level (int, optional): The depth of pages to download from the url_prefix. Defaults to 3.
            output_dir (str, optional): Where to output. It will create if it does not exist. Defaults to "~/outputs/".
            meta_file (str, optional): Where to output the url to uuid filename pairs (csv). Defaults to "meta.csv".
            get_urls ([type], optional): Optional python function to `get_urls(soup: BeautifulSoup, url="")` function. 
            Defaults to `yadil.web.image_scraper_config.default_config.get_urls`.
        """
        scraper_main(
            url_prefix=url_prefix,
            pg_start=pg_start,
            pg_end=pg_end,
            image_types=image_types,
            max_level=max_level,
            output_dir=output_dir,
            meta_file=meta_file,
            get_urls=get_urls,
        )

    def merge_meta_files(self, input_globs: Union[List, str], output_path: str = None):
        """Merge the meta files (url, uuid) and outputs.

        Args:
            input_globs (Union[List, str]): glob(s) to merge.
            output_path (str, optional): output path. Defaults to None.
        """
        df = merge_meta_files(input_meta_glob=input_globs)
        df.to_csv(output_path, index=False) if output_path else print(df)

    def create_meta_file_from_glob(self, input_globs: Union[List, str], output_path: str = None):
        """Create meta file from glob (with empty url) and outputs.

        Args:
            input_globs (Union[List, str]): glob(s) to read in.
            output_path (str, optional): output_path. Defaults to None.
        """
        df = create_meta_file_from_glob(input_globs=input_globs)
        df.to_csv(output_path, index=False) if output_path else print(df)


if __name__ == "__main__":
    fire.Fire(WebUtils())
