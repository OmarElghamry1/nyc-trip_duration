import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures

from src.data_loader import load_train
from src.preprocessing import add_haversine_distance, add_time_features, preprocess_data, remove_with_hard_filters
from src.utils.cli import parse_arguments


RANDOM_STATE = 314

if __name__ == '__main__':

    parser = parse_arguments()

    args = parser.parse_args()

    print("Preprocessing:", args.preprocessing)
    print("Dataset:", args.dataset)
    print("Model: Ridge Regression (alpha=1)")
    print("Features:", args.features)

    dataset = args.dataset

    if dataset == 1:

        df = load_train()

        df = preprocess_data(df)

        df = remove_with_hard_filters(df,
                                      column_name = 'trip_duration',
                                      low_val=100,
                                      high_val=12600)
        df = remove_with_hard_filters(df,
                                      column_name='trip_duration',
                                      low_only=True,
                                      low_val=0)

        cat_cols = ['month', 'hour', 'day_of_week', 'vendor_id']
        num_cols = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
        num_transformer = MinMaxScaler()

        if args.features in (2, 3, 4, 5, 6):
            print("Feature: Haversine distance")
            df = add_haversine_distance(df)
            num_cols = num_cols + ['haversine_distance']

        if args.features == 3:
            print("Feature: Polynomial (degree 2)")
            num_transformer = Pipeline(steps=[
                ('poly', PolynomialFeatures(degree=2)),
                ('scaler', MinMaxScaler()),
            ])

        if args.features in (4, 5, 6):
            print("Feature: sqrt(Haversine distance)")
            df['haversine_distance_sqrt'] = np.sqrt(df['haversine_distance'])
            num_cols = num_cols + ['haversine_distance_sqrt']

        if args.features in (5, 6):
            print("Feature: Rush hour")
            print("Feature: Weekday")
            df = add_time_features(df)
            num_cols = num_cols + ['is_rush_hour', 'is_weekday']

        X = df[cat_cols + num_cols]
        y = np.log1p(df['trip_duration'])

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=RANDOM_STATE
        )

        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
            ('num', num_transformer, num_cols),
        ])

        model_steps = [('preprocessor', preprocessor)]

        if args.features == 6:
            model_steps.append(('poly', PolynomialFeatures(degree=2)))

        model_steps.append(('ridge', Ridge(alpha=1)))

        model = Pipeline(steps=model_steps)

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("RMSE:", rmse)
        print("R2:", r2)
