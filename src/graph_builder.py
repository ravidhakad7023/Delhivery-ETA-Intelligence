import networkx as nx
import pandas as pd


class GraphBuilder:

    def build_graph(self, corridor_df):

        G = nx.DiGraph()

        for _, row in corridor_df.iterrows():

            G.add_edge(
                row["source_center"],
                row["destination_center"],
                weight=row["mean_delay_ratio"],
                trip_count=row["trip_count"],
                breach_rate=row["sla_breach_rate"]
            )

        return G

    def compute_graph_metrics(self, G):

        metrics_df = pd.DataFrame()

        metrics_df["node"] = list(G.nodes())

        metrics_df["in_degree"] = [
            G.in_degree(n) for n in G.nodes()
        ]

        metrics_df["out_degree"] = [
            G.out_degree(n) for n in G.nodes()
        ]

        betweenness = nx.betweenness_centrality(G)

        pagerank = nx.pagerank(G)

        metrics_df["betweenness"] = (
            metrics_df["node"]
            .map(betweenness)
        )

        metrics_df["pagerank"] = (
            metrics_df["node"]
            .map(pagerank)
        )

        return metrics_df
    
    def merge_graph_features(self,df,graph_metrics_df):
        source_metrics = graph_metrics_df.copy()

        source_metrics = source_metrics.rename(columns={
            "node": "source_center",
            "betweenness": "source_betweenness",
            "pagerank": "source_pagerank",
            "in_degree": "source_in_degree",
            "out_degree": "source_out_degree"
        })

        df = df.merge(
            source_metrics,
            on="source_center",
            how="left"
        )

        return df