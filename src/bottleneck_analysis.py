import pandas as pd


class BottleneckAnalyzer:

    def rank_hubs(self, graph_metrics_df):

        graph_metrics_df = graph_metrics_df.copy()

        graph_metrics_df["bottleneck_score"] = (
            0.4 * graph_metrics_df["betweenness"]
            + 0.3 * graph_metrics_df["pagerank"]
            + 0.3 * graph_metrics_df["out_degree"]
        )

        graph_metrics_df = (
            graph_metrics_df
            .sort_values(
                by="bottleneck_score",
                ascending=False
            )
        )

        return graph_metrics_df

    def rank_corridors(self, corridor_df):

        corridor_df = corridor_df.copy()

        corridor_df["corridor_risk"] = (
            0.4 * corridor_df["mean_delay_ratio"]
            + 0.3 * corridor_df["sla_breach_rate"]
            + 0.3 * corridor_df["delay_variance"]
        )

        corridor_df = (
            corridor_df
            .sort_values(
                by="corridor_risk",
                ascending=False
            )
        )

        return corridor_df