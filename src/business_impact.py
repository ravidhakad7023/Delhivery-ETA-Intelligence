import pandas as pd


class BusinessImpact:

    def __init__(self):
        pass

    def estimate_revenue_risk(
        self,
        corridor_df
    ):

        corridor_df["estimated_revenue_risk"] = (
            corridor_df["sla_breach_rate"]
            *
            corridor_df["trip_count"]
            *
            500
        )

        return corridor_df

    def estimate_hub_impact(
        self,
        top_hubs
    ):

        top_hubs["estimated_delay_cost"] = (
            top_hubs["bottleneck_score"]
            * 10000
        )

        return top_hubs