import sys
from unittest.suite import TestSuite

if "." not in sys.path:
    sys.pah.append(".")

import pathlib
import threading
import glob
from tempfile import TemporaryDirectory
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from unittest import TestCase

from yadil.web.scraper import _main, main
from yadil.web.scraper_config import default_config


def _start_server(server: ThreadingHTTPServer):
    server.serve_forever(poll_interval=1)


THIS_FILE_PATH = pathlib.Path(__file__)
THIS_DIRECTORY = THIS_FILE_PATH.parent
SITE_DIRECTORY = pathlib.Path.joinpath(THIS_DIRECTORY, "sample-site")


class MyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SITE_DIRECTORY, **kwargs)


server = ThreadingHTTPServer(server_address=("", 8080), RequestHandlerClass=(MyHandler))
t = threading.Thread(target=server.serve_forever, args=(1,))


class TestSampleSiteLegacy(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        t.start()
        super().setUpClass()
        
    @classmethod
    def tearDownClass(cls) -> None:
        server.shutdown()
        t.join()
        # self.p.join()
        return super().tearDownClass()
    
    def test_legacy(self) -> None:
        with TemporaryDirectory() as output_dir:
            default_config.OUTPUT_DIR = output_dir
            default_config.DELAY_MEAN = 0.0
            default_config.DELAY_STD = 0.0
            _main(config=default_config)
            files = glob.glob(output_dir + "/*")
            assert len(files) == 10

    def test_main(self) -> None:
        with TemporaryDirectory() as output_dir:
            main(output_dir=output_dir, meta_file="meta_2.csv")
            files =glob.glob(output_dir + "/*")
            assert len(files) == 10
