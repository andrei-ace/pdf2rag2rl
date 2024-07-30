import numpy as np
import torch


def get_possible_node_pairs(x, edge_index):
    node_count = x.size(0)
    all_pairs = [(i, j) for i in range(node_count) for j in range(node_count) if i != j]
    existing_edges = set(tuple(edge) for edge in edge_index.t().tolist())
    possible_add_pairs = [pair for pair in all_pairs if pair not in existing_edges]
    possible_remove_pairs = [pair for pair in all_pairs if pair in existing_edges]
    return possible_add_pairs, possible_remove_pairs


def apply_action(graph, edges, action_name, node_pair):
    if action_name == "add":
        # Add edge to graph
        node1, node2 = node_pair
        graph.edge_index = torch.cat([graph.edge_index, torch.tensor([[node1, node2], [node2, node1]])], dim=1)
        edges.append((node1, node2))
        edges.append((node2, node1))
    elif action_name == "remove":
        # Remove edge from graph
        node1, node2 = node_pair
        mask = ~(
            torch.logical_and(graph.edge_index[0] == node1, graph.edge_index[1] == node2)
            | torch.logical_and(graph.edge_index[0] == node2, graph.edge_index[1] == node1)
        )
        graph.edge_index = graph.edge_index[:, mask]
        edges.remove((node1, node2))
        edges.remove((node2, node1))

    return graph, edges


def revert_action(graph, edges, action_name, node_pair):
    if action_name == "add":
        # Remove the previously added edge
        node1, node2 = node_pair
        mask = ~(
            torch.logical_and(graph.edge_index[0] == node1, graph.edge_index[1] == node2)
            | torch.logical_and(graph.edge_index[0] == node2, graph.edge_index[1] == node1)
        )
        graph.edge_index = graph.edge_index[:, mask]
        edges.remove((node1, node2))
        edges.remove((node2, node1))
    elif action_name == "remove":
        # Add the previously removed edge
        node1, node2 = node_pair
        graph.edge_index = torch.cat([graph.edge_index, torch.tensor([[node1, node2], [node2, node1]])], dim=1)
        edges.append((node1, node2))
        edges.append((node2, node1))

    return graph, edges


# Example usage
def sample_action(graph, add_net, remove_net, critic_net, action_probs):
    # Sample an action based on policy output
    actions = ["add", "remove", "stop"]
    chosen_action_name = np.random.choice(actions, p=action_probs.squeeze().detach().numpy())

    possible_add_pairs, possible_remove_pairs = get_possible_node_pairs(graph.x, graph.edge_index)

    if chosen_action_name == "add" and possible_add_pairs:
        add_probs = []
        for pair in possible_add_pairs:
            node1_emb = graph.x[pair[0]].unsqueeze(0)
            node2_emb = graph.x[pair[1]].unsqueeze(0)
            add_prob = add_net(node1_emb, node2_emb)
            add_probs.append(add_prob.item())
        add_probs = torch.tensor(add_probs)
        add_probs = torch.nn.functional.softmax(add_probs, dim=0)
        chosen_idx = np.random.choice(len(possible_add_pairs), p=add_probs.squeeze().detach().numpy())
        chosen_action = possible_add_pairs[chosen_idx]
        chosen_prob = action_probs[:, 0]
        # Compute and print critic value
        node1_emb = graph.x[chosen_action[0]].unsqueeze(0)
        node2_emb = graph.x[chosen_action[1]].unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            torch.tensor([1, 0, 0]).unsqueeze(0),
            node1_emb,
            node2_emb,
            chosen_prob.unsqueeze(0),
        )
    elif chosen_action_name == "remove" and possible_remove_pairs:
        remove_probs = []
        for pair in possible_remove_pairs:
            node1_emb = graph.x[pair[0]].unsqueeze(0)
            node2_emb = graph.x[pair[1]].unsqueeze(0)
            remove_prob = remove_net(node1_emb, node2_emb)
            remove_probs.append(remove_prob.item())
        remove_probs = torch.tensor(remove_probs)
        remove_probs = torch.nn.functional.softmax(remove_probs, dim=0)
        chosen_idx = np.random.choice(len(possible_remove_pairs), p=remove_probs.squeeze().detach().numpy())
        chosen_action = possible_remove_pairs[chosen_idx]
        chosen_prob = action_probs[:, 1]
        # Compute and print critic value
        node1_emb = graph.x[chosen_action[0]].unsqueeze(0)
        node2_emb = graph.x[chosen_action[1]].unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            torch.tensor([0, 1, 0]).unsqueeze(0),
            node1_emb,
            node2_emb,
            chosen_prob.unsqueeze(0),
        )
    else:
        chosen_action_name = "stop"
        chosen_action = None
        chosen_prob = action_probs[:, 2]
        # Compute and print critic value
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            torch.tensor([0, 0, 1]).unsqueeze(0),
            graph.x.mean(dim=0).unsqueeze(0),
            graph.x.mean(dim=0).unsqueeze(0),
            chosen_prob.unsqueeze(0),
        )

    return chosen_action_name, chosen_action, chosen_prob, critic_value
