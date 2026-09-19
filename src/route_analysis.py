import pandas as pd


class RouteAnalysis:

    def __init__(self):
        pass

    def compare_route_types(self, df):

        analysis = (
            df.groupby(
                "recommended_route_type"
            )
            .agg({
                "delay_ratio": [
                    "mean",
                    "median"
                ],

                "sla_breach": "mean",

                "osrm_distance": "mean",

                "trip_uuid": "count",

                "route_risk_score": "mean"
            })
        )

        return analysis

    def corridor_level_analysis(self, df):

        corridor_analysis = (
            df.groupby([
                "source_center",
                "destination_center",
                "recommended_route_type"
            ])
            .agg({
                "delay_ratio": "mean",
                "sla_breach": "mean",
                "trip_uuid": "count"
            })
            .reset_index()
        )

        corridor_analysis = (
            corridor_analysis
            .sort_values(
                by="delay_ratio",
                ascending=False
            )
        )

        return corridor_analysis