# yadil - Yet Another Document and Image Library

## web

### image_scraper

Via CLI:

```bash
pip install yadil
python -m yadil.web.image_scraper --help
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

## Copy-rights etc.

The test images in tests/sample-site retrieved from the internet are attributed as various `Creative Common` license, some of them have commercial use prohibited. 

The [test image](https://live.staticflickr.com/2605/3721476240_bf643c709e.jpg) retrived from https://live.staticflickr.com is attributed as follow:

> "Models photo shoot" by davidyuweb is licensed under CC BY-NC 2.0

The [test image](https://www.pexels.com/photo/back-view-of-a-woman-in-brown-dress-3866555/) is attributed as follow:

> All photos and videos on Pexels are free to use.
> Attribution is not required. Giving credit to the photographer or Pexels is not necessary but always appreciated.
> You can modify the photos and videos from Pexels. Be creative and edit them as you like. 
> by [Pexels.com's license](https://www.pexels.com/license/)

The [test image](https://unsplash.com/photos/6xv4A1VA1rU) is attributed as follow:

> All photos can be downloaded and used for free
> Commercial and non-commercial purposes
> No permission needed (though attribution is appreciated!)
> What is not permitted 👎
> Photos cannot be sold without significant modification.
> Compiling photos from Unsplash to replicate a similar or competing service.
> by [Unsplash.com's license](https://unsplash.com/license)

The [insightface's code](https://pypi.org/project/insightface/) is attributed as follow:

> MIT License 

The [insightface's models](https://pypi.org/project/insightface/), which the above code automatically downloads is governed by: 

> Non commercial License

For more information, refer to [insightface github page](https://github.com/deepinsight/insightface). 