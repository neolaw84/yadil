# yadil - Yet Another Document and Image Library

## web

### image_scraper

Via CLI:

```bash
pip install yadil
python -m yadil.web.image_scraper --help
NAME
    image_scraper.py - download image files from a series of web-pages into output_dir.

SYNOPSIS
    image_scraper.py <flags>

DESCRIPTION
    Note: press 'q' to quit from `less`-like or `vim`-like environment.

FLAGS
    --url_prefix=URL_PREFIX
        Type: str
        Default: 'http://local...
        Prefix of the urls to get series of web-pages. You need to include `{page-id}`
    --pg_start=PG_START
        Type: int
        Default: 1
        Starting page to be placed in `{page-id}` placeholder. Defaults to 1.
    --pg_end=PG_END
        Type: int
        Default: 2
        Ending page to be placed in `{page-id}` placeholder. Defaults to 2.
    --image_types=IMAGE_TYPES
        Type: typing.List
        Default: ['image/jpe...
        A list of meme-types to download. Defaults to ["image/jpeg", "image/png", "image/jpg"].
    --max_level=MAX_LEVEL
        Default: 3
        The depth of pages to download from the url_prefix. Defaults to 3.
    --output_dir=OUTPUT_DIR
        Default: '~/outputs/'
        Where to output. It will create if it does not exist. Defaults to "~/outputs/".
    --meta_file=META_FILE
        Default: 'meta.csv'
        Where to output the url to uuid filename pairs (csv). Defaults to "meta.csv".
    --get_urls=GET_URLS
        Type: typing.Union[str, typin...
        Default: <function get_...
        Optional python function to `get_urls(soup: BeautifulSoup, url="")` function. Defaults to `yadil.web.image_scraper_config.default_config.get_urls`.
```

Via python: 

```
from yadil.web.image_scraper import main as is_main

is_main(
    url_prefix: str = "http://localhost:8080/page-{page_id}",
    pg_start: int = 1,
    pg_end: int = 2,
    image_types: List = ["image/jpeg", "image/png", "image/jpg"],
    max_level=3,
    output_dir="~/outputs/",
    meta_file="meta.csv",
)
```