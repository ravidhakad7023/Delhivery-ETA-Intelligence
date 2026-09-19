import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D


class NetworkVisualizer:

    def __init__(self):
        pass

    def visualize_network(
        self,
        G,
        graph_metrics_df,
        top_corridors
    ):

        # =====================================
        # REMOVE SELF LOOPS
        # =====================================

        G.remove_edges_from(
            nx.selfloop_edges(G)
        )

        # =====================================
        # TOP IMPORTANT NODES
        # =====================================

        top_nodes = (
            graph_metrics_df
            .sort_values(
                by="betweenness",
                ascending=False
            )
            .head(40)["node"]
            .tolist()
        )

        subgraph = G.subgraph(top_nodes).copy()

        # =====================================
        # IDENTIFY HIGH-RISK CORRIDORS
        # =====================================

        risky_edges = set()

        for _, row in top_corridors.iterrows():

            source = row["source_center"]
            destination = row["destination_center"]

            risky_edges.add(
                (source, destination)
            )

        # =====================================
        # LAYOUT
        # =====================================

        plt.figure(figsize=(20, 14))

        pos = nx.spring_layout(
            subgraph,
            k=1.2,
            iterations=100,
            seed=42
        )

        # =====================================
        # NODE SIZE
        # =====================================

        metric_map = (
            graph_metrics_df
            .set_index("node")["betweenness"]
            .to_dict()
        )

        node_sizes = []

        for node in subgraph.nodes():

            val = metric_map.get(node, 0)

            node_sizes.append(
                300 + (val * 8000)
            )

        # =====================================
        # DRAW NODES
        # =====================================

        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_size=node_sizes,
            alpha=0.85
        )

        # =====================================
        # DRAW EDGES
        # =====================================

        normal_edges = []
        risky_edges_present = []

        for edge in subgraph.edges():

            if edge in risky_edges:

                risky_edges_present.append(edge)

            else:

                normal_edges.append(edge)

        # NORMAL EDGES

        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=normal_edges,
            edge_color="gray",
            alpha=0.2,
            arrows=True,
            arrowsize=10
        )

        # RISKY EDGES

        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=risky_edges_present,
            edge_color="red",
            width=2.5,
            alpha=0.8,
            arrows=True,
            arrowsize=12
        )

        # =====================================
        # LABEL TOP HUBS
        # =====================================

        top_40 = set(top_nodes[:40])

        labels = {}

        for i, node in enumerate(top_40):

            labels[node] = f"H{i+1}"

        nx.draw_networkx_labels(
            subgraph,
            pos,
            labels=labels,
            font_size=10
        )

        # =====================================
        # LEGEND
        # =====================================

        legend_elements = [

            Line2D(
                [0],
                [0],
                color="red",
                lw=3,
                label="High Delay Corridors"
            ),

            Line2D(
                [0],
                [0],
                color="gray",
                lw=2,
                label="Normal Corridors"
            )
        ]

        plt.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=12
        )

        # =====================================
        # TITLE
        # =====================================

        plt.title(
            "Logistics Network Bottlenecks & Delay Corridors",
            fontsize=20
        )

        plt.axis("off")

        plt.tight_layout()

        # =====================================
        # PRINT HUB LABELS
        # =====================================

        print("\nBOTTLENECK HUB LABELS\n")

        for i, node in enumerate(top_40):

            print(f"H{i+1} --> Hub {int(node)}")

        plt.show()