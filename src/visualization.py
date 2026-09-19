import matplotlib.pyplot as plt
import networkx as nx


class Visualizer:

    def plot_network_graph(self, G):

        plt.figure(figsize=(12, 8))

        pos = nx.spring_layout(G)

        nx.draw(
            G,
            pos,
            with_labels=False,
            node_size=50,
            arrows=True
        )

        plt.title("Logistics Network Graph")

        plt.show()

    def plot_prediction_scatter(self, y_true, y_pred):

        plt.figure(figsize=(8, 6))

        plt.scatter(y_true, y_pred, alpha=0.5)

        plt.xlabel("Actual")
        plt.ylabel("Predicted")

        plt.title("Predicted vs Actual")

        plt.show()