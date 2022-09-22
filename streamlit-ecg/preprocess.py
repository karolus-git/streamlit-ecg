
import multiprocessing
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import utils
import requests
import numpy as np
import scipy
from scipy.ndimage import zoom
from pandarallel import pandarallel
import time
import matplotlib.pyplot as plt
from scipy import signal
from skimage.restoration import denoise_wavelet
import wfdb
import glob
load_dotenv()

PATH = os.getenv("DB_PATH")
RESAMPLING = int(os.getenv("RESAMPLING"))
WORKERS = int(os.getenv("WORKERS"))
FILTER = int(os.getenv("FILTER"))
CACHE = os.getenv("CACHE")

def download_dataset():
    wfdb.dl_database('mitdb', PATH)


def save_dataset(df):

    if not Path(CACHE).exists():
        Path(CACHE).mkdir()

    df.to_pickle(Path(CACHE, "data.pkl"))

def load_dataset():
    return pd.read_pickle(Path(CACHE, "data.pkl"))

def build_dataset(autosave=True):

    start_time = time.time()

    if not glob.glob(PATH):
        download_dataset()
        
    #FInd out if all files are are
    records = utils.read_records(PATH)

    #Build the paths of those records
    paths = [str(Path(PATH, record)) for record in records]

    #Create a Pool
    pool_obj = multiprocessing.Pool()

    #For each path, extract the record and the beats into a dataframe
    utils.extract_records(paths[0])
    dfs = pool_obj.map(utils.extract_records, paths)

    #Merge the dataframesfrom skimage.restoration import denoise_wavelet
    df = pd.concat(dfs)

    #Apply resampling of the signals
    df = filter_dataset(df, apply_filter=FILTER, workers=WORKERS)

    print(f"build done in {time.time() - start_time:02.2}s")

    #Save it
    if autosave:
        save_dataset(df)
    
    return df

def transform_datatset(drop_columns=["bpm", "id", "age", "gender", "fs"]):

    df = df.drop(drop_columns, axis=1)

    return df

def filter_dataset(df, workers=3, apply_filter=True, apply_resample=True):

    def scipy_resample(y):
        return scipy.signal.resample(y, RESAMPLING)

    def pywt_wavlet(y):
        return denoise_wavelet(y, 
            method='BayesShrink', 
            mode='soft', 
            wavelet_levels=2, 
            wavelet='sym8', 
            rescale_sigma='True')

    def actions(row):
        y = row.beat
        if apply_filter:
            y = pywt_wavlet(y)
        if apply_resample:
            y = scipy_resample(y)
        return y
    
    pandarallel.initialize(
        progress_bar=True,
        nb_workers=workers)

    # Apply filter and resampling
    df["beat"] = df.parallel_apply(actions, axis=1)

    return df