import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):

    df.drop(columns=["id"], inplace=True)
    df.insert(0, "year", df["pickup_datetime"].dt.year)
    df.insert(1, "month", df["pickup_datetime"].dt.month)
    df.insert(2, "hour", df["pickup_datetime"].dt.hour)
    df.insert(3, "day_of_week", df["pickup_datetime"].dt.dayofweek)

    df['vendor_id'] = df['vendor_id'].astype('category')

    df.drop(columns=['year', 'pickup_datetime', "store_and_fwd_flag"], inplace=True)

    return df

def remove_with_hard_filters(df, column_name, low_val, high_val=None, low_only=False, high_only=False):
    if low_only:
        df = df[df[column_name] > low_val]
        return df

    elif high_only:
        df = df[df[column_name] < high_val]
        return df
    else:
        return df[(df[column_name] > low_val) & (df[column_name] < high_val)]


def add_haversine_distance(df, lat1='pickup_latitude', lon1='pickup_longitude',
                             lat2='dropoff_latitude', lon2='dropoff_longitude',
                             col_name='haversine_distance'):
    """
    Add great-circle distance (in km) between two lat/lon points.
    Uses the haversine formula.
    """
    df = df.copy()

    R = 6371  # Earth's radius in km

    lat1_r = np.radians(df[lat1])
    lat2_r = np.radians(df[lat2])
    dlat = np.radians(df[lat2] - df[lat1])
    dlon = np.radians(df[lon2] - df[lon1])

    a = np.sin(dlat / 2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    df[col_name] = R * c

    return df

def add_time_features(df, day_of_week_start='monday'):
    """
    Add is_rush_hour, is_weekday, is_holiday based on hour/day_of_week/month.
    day_of_week_start: 'monday' (0=Mon...6=Sun) or 'sunday' (0=Sun...6=Sat)
    """
    df = df.copy()

    # Rush hour: 7-9am and 4-7pm
    df['is_rush_hour'] = (df['hour'].between(14, 16)).astype(int)

    # Weekday: depends on day_of_week encoding
    if day_of_week_start == 'monday':
        weekend_days = [5, 6]  # Sat, Sun
    elif day_of_week_start == 'sunday':
        weekend_days = [0, 6]  # Sun, Sat
    else:
        raise ValueError("day_of_week_start must be 'monday' or 'sunday'")

    df['is_weekday'] = (~df['day_of_week'].isin(weekend_days)).astype(int)

    return df

def get_scaler(preprocessing):
    if preprocessing == 0:
        return "passthrough"
    if preprocessing == 1:
        return MinMaxScaler()
    if preprocessing == 2:
        return StandardScaler()

    raise ValueError("Invalid preprocessing option")

def make_train_test_split(df, target='trip_duration', test_size=0.1, random_state=314, log_target=True):
    """
    Split df into X_train, X_test, y_train, y_test.
    X is every column except target. y is target (optionally log1p-transformed).
    """

    X = df.drop(columns=[target])
    y = np.log1p(df[target]) if log_target else df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test

def one_hot_encode(df, cat_cols, drop_first=True):
    """
    One-hot encode true categorical columns (no inherent order): vendor_id,
    day_of_week, month, hour, etc. drop_first avoids the dummy-variable trap
    (perfect multicollinearity with the intercept).
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)

    return df