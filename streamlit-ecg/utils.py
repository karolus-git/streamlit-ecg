import os
from dotenv import load_dotenv
import numpy as np
import wfdb
import pandas as pd
import random
import glob
from pathlib import Path

load_dotenv()

MAX_BPM = int(os.getenv("MAX_BPM"))
MIN_BPM = int(os.getenv("MIN_BPM"))
BEAT_INVALID = [
    "[", "!", "]", "x", "(", ")", "p", "t", 
    "u", "`", "'", "^", "|", "~", "+", "s", 
    "T", "*", "D", "=",'"', "@", "B", "a", "J", "S",
    "r", "F", "e", "j", "n", "f", "Q", "?"
]

def read_records(path):
    #Read the indexes of records stored in that folder

    dat = {Path(f).stem for f in glob.glob(path+"/*.dat")}
    hea = {Path(f).stem for f in glob.glob(path+"/*.hea")}
    atr = {Path(f).stem for f in glob.glob(path+"/*.atr")}


    return dat.intersection(hea).intersection(atr)

def extract_records(path):

    #Get data and annotations
    data = wfdb.rdrecord(path)
    annotations = wfdb.rdann(path, 'atr')

    #Create a mask for valid beats
    mask = np.isin(annotations.symbol, BEAT_INVALID, invert=True)

    # Get channel, symbols, samples, fs, and infos
    channel = 1
    symbols = np.array(annotations.symbol)[mask]
    samples = np.array(annotations.sample)[mask]
    signal = data.p_signal
    fs = annotations.fs
    infos = data.comments[0].split()

    # For this record, split the signal into beats and push them into a df
    df = extract_beats(signal, samples, symbols, fs, channel)

    # Some info
    df["gender"] = str(infos[1])
    df["age"] = int(infos[0])
    df["fs"] = int(fs)
    
    return df

def extract_beats(signal, samples, symbols, fs, channel):

    beats = {
        "bpm" : [],
        "symbol" : [],
        "beat" : [],
        "bid" : [],
    }
    
    #For each pair of sample
    for sample, next_sample, symbol in zip(samples, samples[1:], symbols):
        #Calculate the bpm value
        bpm = get_bpm(sample, next_sample, fs)

        bid = random.randint(0, 0xFFFFFF)
        # If that bpm value is in our range
        if MAX_BPM > bpm > MIN_BPM :
            
            #We extract the signal
            beat = signal[sample:next_sample, channel]

            # And store it and other infos in the beats dict
            beats["beat"].append(beat)
            beats["bpm"].append(bpm)
            beats["symbol"].append(symbol)
            beats["bid"].append(bid)

    return pd.DataFrame.from_dict(beats)

def get_bpm(sample, next_sample, fs):

    frames = next_sample - sample
    duration = frames / fs

    return round(60 / duration)