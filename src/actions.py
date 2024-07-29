import numpy as np
import torch
import torch.nn.functional as F


def get_possible_node_pairs(x, edge_index):
    node_count = x.size(0)
    all_pairs = [(i, j) for i in range(node_count) for j in range(node_count) if i != j]
    existing_edges = set(tuple(edge) for edge in edge_index.t().tolist())
    possible_add_pairs = [pair for pair in all_pairs if pair not in existing_edges]
    possible_remove_pairs = [pair for pair in all_pairs if pair in existing_edges]
    return possible_add_pairs, possible_remove_pairs


def sample_action(graph, add_net, remove_net, critic_net, action_probs):

    # Sample an action based on the policy network output
    actions = ["add", "remove", "stop"]
    action_probs = action_probs.squeeze().detach().numpy()
    chosen_action = np.random.choice(actions, p=action_probs)
    print(f"Chosen action: {chosen_action}")

    # Create lists of possible node pairs for add and remove action# Create lists of possible node pairs for add and remove actions
    possible_add_pairs, possible_remove_pairs = get_possible_node_pairs(
        graph.x, graph.edge_index
    )

    if chosen_action == "add":
        # Calculate add probabilities for possible node pairs
        add_probs = []
        for pair in possible_add_pairs:
            node1_emb = graph.x[pair[0]].unsqueeze(0)
            node2_emb = graph.x[pair[1]].unsqueeze(0)
            add_prob = add_net(node1_emb, node2_emb)
            add_probs.append(add_prob.item())
        add_probs = torch.tensor(add_probs)
        add_probs = F.softmax(add_probs, dim=0)

        # Sample a specific add action
        add_action_idx = np.random.choice(
            len(possible_add_pairs), p=add_probs.detach().numpy()
        )
        add_action = possible_add_pairs[add_action_idx]
        print(f"Add edge between nodes: {add_action}")

        # Compute and print critic value
        node1_emb = graph.x[add_action[0]].unsqueeze(0)
        node2_emb = graph.x[add_action[1]].unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            torch.tensor([1, 0, 0]).unsqueeze(0),
            node1_emb,
            node2_emb,
            torch.tensor([action_probs[0]]).unsqueeze(0),
        )
        print(f"Critic value: {critic_value.item()}")

    elif chosen_action == "remove":
        # Calculate remove probabilities for possible node pairs
        remove_probs = []
        for pair in possible_remove_pairs:
            node1_emb = graph.x[pair[0]].unsqueeze(0)
            node2_emb = graph.x[pair[1]].unsqueeze(0)
            remove_prob = remove_net(node1_emb, node2_emb)
            remove_probs.append(remove_prob.item())
        remove_probs = torch.tensor(remove_probs)
        remove_probs = F.softmax(remove_probs, dim=0)

        # Sample a specific remove action
        remove_action_idx = np.random.choice(
            len(possible_remove_pairs), p=remove_probs.detach().numpy()
        )
        remove_action = possible_remove_pairs[remove_action_idx]
        print(f"Remove edge between nodes: {remove_action}")

        # Compute and print critic value
        node1_emb = graph.x[remove_action[0]].unsqueeze(0)
        node2_emb = graph.x[remove_action[1]].unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            torch.tensor([0, 1, 0]).unsqueeze(0),
            node1_emb,
            node2_emb,
            torch.tensor([action_probs[1]]).unsqueeze(0),
        )
        print(f"Critic value: {critic_value.item()}")

    else:
        print("Stop modifying the graph")

        # Compute and print critic value
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            torch.tensor([0, 0, 1]).unsqueeze(0),
            graph.x.mean(dim=0).unsqueeze(0),
            graph.x.mean(dim=0).unsqueeze(0),
            torch.tensor([action_probs[2]]).unsqueeze(0),
        )
        print(f"Critic value: {critic_value.item()}")
