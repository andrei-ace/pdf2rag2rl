import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# Define the neural network models for actions
class AddNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AddNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node1_emb, node2_emb):
        x = torch.cat([node1_emb, node2_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class RemoveNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RemoveNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node1_emb, node2_emb):
        x = torch.cat([node1_emb, node2_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 3)  # Three actions: add, remove, or stop

    def forward(self, x, edge_index):
        # GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = torch.mean(x, dim=0, keepdim=True)  # Pooling node embeddings

        # Fully connected layer to output action probabilities
        x = self.fc(x)
        return F.softmax(x, dim=-1)


# Define the critic network
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, projection_dim):
        super(CriticNetwork, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.node_projection = nn.Linear(input_dim, projection_dim)
        # GCN output, projected node1_emb, node2_emb, action one-hot vector, and action probability
        self.fc1 = nn.Linear(hidden_dim + 2 * projection_dim + 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, action_one_hot, node1_emb, node2_emb, action_prob):
        # GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        gcn_output = torch.mean(x, dim=0, keepdim=True)  # Pooling node embeddings

        # Project node embeddings
        node1_emb_proj = F.relu(self.node_projection(node1_emb))
        node2_emb_proj = F.relu(self.node_projection(node2_emb))

        # Concatenate GCN output, projected node embeddings, action one-hot vector, and action probability
        concatenated_input = torch.cat(
            [gcn_output, node1_emb_proj, node2_emb_proj, action_one_hot, action_prob],
            dim=-1,
        )

        # Fully connected layers
        x = F.relu(self.fc1(concatenated_input))
        x = self.fc2(x)
        return x
