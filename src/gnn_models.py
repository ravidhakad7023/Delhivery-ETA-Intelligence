import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x


class GNNTrainer:

    def train_model(
        self,
        model,
        graph_data,
        epochs=100,
        lr=0.01
    ):

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        model.train()

        for epoch in range(epochs):

            optimizer.zero_grad()

            out = model(
                graph_data.x,
                graph_data.edge_index
            )

            loss = F.mse_loss(
                out.squeeze(),
                graph_data.y
            )

            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.4f}"
                )