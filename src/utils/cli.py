import argparse

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preprocessing",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="0: none, 1: MinMaxScaler, 2: StandardScaler",
    )

    parser.add_argument(
        "--dataset",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="1: full, 2: sample, 3: validation, 4: test",
    )

    parser.add_argument(
        "--model",
        type=int,
        choices=[1],
        default=1,
        help="1: ridge",
    )

    parser.add_argument(
        "-f", "--features",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help="1: none, 2: haversine distance, 3: haversine + degree-2 polynomial features, "
             "4: haversine + sqrt(haversine distance), 5: haversine + sqrt(haversine) + rush hour + weekday, "
             "6: same as 5 + degree-2 polynomial features after one-hot encoding",
    )

    return parser


