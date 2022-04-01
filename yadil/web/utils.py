import os
import glob
import pathlib
from typing import List, Union

import pandas as pd
from pandas.core.algorithms import isin


def merge_meta_files(input_meta_glob: Union[List, str]) -> pd.DataFrame:
    def read_df(meta_fp):
        df = pd.read_csv(meta_fp, names=["url", "uuid"])
        df = df[(df.url != "url")]
        return df

    input_meta_globs = [input_meta_glob] if isinstance(input_meta_glob, str) else input_meta_glob
    dfs = [read_df(meta_fp) for img in input_meta_globs for meta_fp in glob.glob(img)]
    df_all = pd.concat(dfs)
    df_all.sort_values(by=["url", "uuid"], inplace=True, na_position="last", ignore_index=True)
    df_all.drop_duplicates(inplace=True, keep="first")

    # df_all = df_all.set_index("uuid")
    return df_all


def create_meta_file_from_glob(input_globs: Union[List, str],):
    def remove_ext(f):
        try:
            p = pathlib.Path(f)
            return "_".join(p.name.split(".")[:-1])
        except:
            return ""

    gls = [input_globs] if isinstance(input_globs, str) else input_globs
    df = pd.DataFrame({"url": "", "uuid": [remove_ext(f) for gl in gls for f in glob.glob(gl)]})
    df.drop_duplicates(inplace=True)
    return df
