import pandas as pd
import numpy as np


class RouteStrategy:

    def __init__(self):
        pass

    def compute_route_risk_score(self, df):

        df["route_risk_score"] = (
            0.4 * df["source_betweenness"]
            + 0.3 * df["source_pagerank"]
            + 0.3 * df["delay_ratio"]
        )

        return df

    def recommend_route_type(self, df):

        conditions = [
            (
                (df["route_risk_score"] > 1.0)
                &
                (df["osrm_distance"] > 300)
            ),

            (
                (df["route_risk_score"] <= 1.0)
            )
        ]

        choices = [
            "FTL",
            "Carting"
        ]

        df["recommended_route_type"] = np.select(
            conditions,
            choices,
            default="Carting"
        )

        return df

    def route_strategy_summary(self, df):

        summary = (
            df.groupby(
                "recommended_route_type"
            )
            .agg({
                "delay_ratio": "mean",
                "sla_breach": "mean",
                "trip_uuid": "count"
            })
            .reset_index()
        )

        return summary