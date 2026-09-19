import pandas as pd
import numpy as np


class FeatureEngineer:

    def create_temporal_features(self, df):

        df = df.copy()

        timestamp_col = "od_start_time"

        df["hour"] = df[timestamp_col].dt.hour
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        df["month"] = df[timestamp_col].dt.month

        df["is_weekend"] = (
            df["day_of_week"].isin([5, 6])
        ).astype(int)

        peak_hours = [8, 9, 10, 17, 18, 19]

        df["is_peak_hour"] = (
            df["hour"].isin(peak_hours)
        ).astype(int)

        return df

    def create_targets(self, df):

        df = df.copy()

        df["delay_ratio"] = (
            df["actual_time"] / df["osrm_time"]
        )

        df["sla_breach"] = np.where(
            df["delay_ratio"] > 1.15,
            1,
            0
        )

        return df

    def build_corridor_features(self, df):

        corridor_stats = (
            df.groupby([
                "source_center",
                "destination_center"
            ])
            .agg(
                mean_delay_ratio=("delay_ratio", "mean"),
                delay_variance=("delay_ratio", "var"),
                avg_distance=("osrm_distance", "mean"),
                avg_actual_time=("actual_time", "mean"),
                sla_breach_rate=("sla_breach", "mean"),
                trip_count=("trip_uuid", "count")
            )
            .reset_index()
        )

        return corridor_stats