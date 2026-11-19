import pandas as pd
from pathlib import Path

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

def load_train() -> DataFrame:
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "train.csv"

    df = pd.read_csv(path, parse_dates=["pickup_datetime"])

    return df