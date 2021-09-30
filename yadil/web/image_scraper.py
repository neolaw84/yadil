import os
import time
import uuid
from uuid import uuid5
import logging
from importlib import import_module
from typing import List, Union, Callable, Tuple

import numpy as np

import fire
import requests
import pandas as pd
from bs4 import BeautifulSoup

from yadil.web.image_scraper_config import default_config as config


def delay_mean_and_std(mean: int = 5, std: int = 3):
    if not delay_mean_and_std.init:
        np.random.seed(delay_mean_and_std.seed)
        delay_mean_and_std.init = True
    time.sleep(np.abs(np.random.normal(loc=mean, scale=std) / 1.0))


delay_mean_and_std.seed = 123456
delay_mean_and_std.init = False

class VisitedUrls(object):

    def __init__(self):
        self.urls = {}

    def check_if_visited_and_add(self, url: str = None):
        try:
            url_parts = url.split("/")
            urls = self.urls
            for p in url_parts:
                if p in urls.keys():
                    urls = urls[p]
                elif p == url_parts[-1]:
                    urls[p] = True
                    return False
                else:
                    urls[p] = {}
                    urls = urls[p]
            return True
        except:
            return False


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _download(config, url) -> Tuple[str, bytes]:
    try:
        if config.visited.check_if_visited_and_add(url):
            logger.warn("url : {} is already visited.".format(url))
            return None
        r = requests.get(url, allow_redirects=True)
        delay_mean_and_std(config.DELAY_MEAN, config.DELAY_STD)
        return r.headers["content-type"], r.content
    except Exception as e:
        logger.error("download error : {}".format(url))
        return None


def _save_image(config, url, content):
    try:
        if content:
            unq_id = str(uuid5(uuid.NAMESPACE_URL, name=url))
            filename = os.path.join(config.OUTPUT_DIR, unq_id + ".jpg")
            with open(filename, "wb") as f:
                f.write(content)
            logger.info("Saving {} as {}".format(url, unq_id))
            df = pd.DataFrame(data={"url": [url], "uuid": [unq_id]}, columns=["url", "uuid"])
            df.to_csv(config.OUTPUT_DIR + "/" + config.META_FILE, index=False, header=False, mode="a")
    except Exception as e:
        logger.error("error with url : {}".format(url))


def visit_page(config, page_url, cur_level=1):
    logger.debug("visiting ... {} and {}".format(page_url, cur_level))
    try:
        ctype, content = _download(config, url=page_url)
        if not content:
            return
        if ctype in config.IMAGE_TYPES:
            # store this image
            _save_image(config, url=page_url, content=content)
        elif ctype.startswith("text/html") and cur_level <= config.MAX_LEVEL:
            soup = BeautifulSoup(content, "html.parser")
            urls = config.get_urls(soup, url=page_url)
            for url in urls:
                visit_page(config, url, cur_level=cur_level + 1)
    except:
        logger.error("Unable to process {url}".format(url=page_url))


def _main(config):
    config.visited = VisitedUrls()
    meta_file_path = os.path.join(config.OUTPUT_DIR, config.META_FILE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if os.path.isfile(meta_file_path):
        df = pd.read_csv(meta_file_path, names=["url", "uuid"])
        _ = [check_if_visited_and_add(u) for u in df.url]
    else:
        df = pd.DataFrame(data={}, columns=["url", "uuid"])
        df.to_csv(meta_file_path, index=False, header=False, mode="a")
    for page_id in range(config.PAGE_START, config.PAGE_END + 1):
        logger.info("working on page_id : {}".format(page_id))
        page_url = config.URL_PREFIX.format(page_id=page_id)
        visit_page(config, page_url, cur_level=1)


def main(
    url_prefix: str = "http://localhost:8080/page-{page_id}",
    pg_start: int = 1,
    pg_end: int = 2,
    image_types: List = ["image/jpeg", "image/png", "image/jpg"],
    max_level=3,
    output_dir="~/outputs/",
    meta_file="meta.csv",
    get_urls:Union[str, Callable]=config.get_urls
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

    if isinstance(get_urls, str):
        get_url_module = get_urls
        path, obj = get_url_module.rsplit(".", 1)
        mod = import_module(path)
        get_urls = getattr(mod, obj)
    config.URL_PREFIX = url_prefix
    config.PAGE_START = pg_start
    config.PAGE_END = pg_end
    config.IMAGE_TYPES = image_types
    config.MAX_LEVEL = max_level
    config.OUTPUT_DIR = output_dir
    config.META_FILE = meta_file
    config.get_urls = get_urls
    _main(config=config)


if __name__ == "__main__":
    fire.Fire(main)
