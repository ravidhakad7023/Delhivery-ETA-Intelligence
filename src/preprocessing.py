import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TIMESTAMP_COLUMNS = [
    "trip_creation_time",
    "od_start_time",
    "od_end_time",
    "cutoff_timestamp"
]


CATEGORICAL_COLUMNS = [
    "route_type",
    "source_center",
    "destination_center"
]


NUMERIC_COLUMNS = [
    "actual_time",
    "osrm_time",
    "factor",
    "segment_actual_time",
    "segment_osrm_time",
    "segment_factor",
    "osrm_distance",
    "segment_osrm_distance",
    "cutoff_factor"
]

class DataPreprocessor:

    def __init__(self, data_path):
        self.data_path = data_path
        self.encoders = {}

    def load_data(self):

        df = pd.read_csv(self.data_path)

        print(f"Loaded dataset shape: {df.shape}")

        return df

    def clean_data(self, df):

        df = df.copy()

        df.drop_duplicates(inplace=True)

        for col in TIMESTAMP_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN")

        df = df[df["actual_time"] > 0]
        df = df[df["osrm_time"] > 0]

        df = df[df["factor"] < 10]

        return df

    def encode_features(self, df):

        df = df.copy()

        for col in CATEGORICAL_COLUMNS:

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(df[col].astype(str))

            self.encoders[col] = encoder

        return df

    def split_data(self, df):

        df = df.sort_values(
            by="od_start_time"
        )

        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))

        train_df = df.iloc[:train_size]

        val_df = df.iloc[
            train_size:train_size + val_size
        ]

        test_df = df.iloc[
            train_size + val_size:
        ]

        return train_df, val_df, test_df