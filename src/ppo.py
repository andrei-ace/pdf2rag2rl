import torch
import torch.nn.functional as F

from actions import apply_action, revert_action, sample_action
from embeddings import EMBEDDING_DIM
from graphs import compute_graph_hash
from models import AddNetwork, CriticNetwork, PolicyNetwork, RemoveNetwork
from rag import rag

MIN_STEPS = 10
MAX_STEPS = 50
EPOCHS = 10

class PPO:
    def __init__(self, device="cpu"):
        input_dim = EMBEDDING_DIM
        hidden_dim = 64
        projection_dim = 32
        # Initialize networks
        self.add_net = AddNetwork(input_dim, hidden_dim)
        self.remove_net = RemoveNetwork(input_dim, hidden_dim)
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        self.critic_net = CriticNetwork(input_dim, hidden_dim, projection_dim)
        self.device = torch.device(device)
        self.add_net.to(self.device)
        self.remove_net.to(self.device)
        self.policy_net.to(self.device)
        self.critic_net.to(self.device)
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.add_net.parameters())
            + list(self.remove_net.parameters())
            + list(self.policy_net.parameters())
            + list(self.critic_net.parameters()),
            lr=1e-3,
        )
        # Initialize loss functions for PPO
        self.value_loss_fn = torch.nn.MSELoss()
        self.policy_loss_fn = torch.nn.CrossEntropyLoss()

    def compute_advantages(self, trajectory, gamma=0.99):
        advantages = []
        returns = []
        G = 0  # Initialize the expected return
        values = [value for _, _, _, value in trajectory]

        # Iterate over the trajectory forward
        for t in range(len(trajectory)):
            G = sum([gamma**i * values[t + i] for i in range(len(values) - t)])
            returns.append(G)

            advantage = G - values[t]
            advantages.append(advantage)

        return advantages, returns

    def collect_trajectory(self, graph, nodes, edges, min_steps=MIN_STEPS, max_steps=MAX_STEPS):
        trajectory = []
        for i in range(max_steps):
            action_probs = self.policy_net(graph.x, graph.edge_index)
            action_name, node_pair, prob, value = sample_action(
                graph, self.add_net, self.remove_net, self.critic_net, action_probs
            )
            if action_name == "stop":
                if i >= min_steps:
                    break
                else:
                    continue
            # save the action, node pair, probability, and value
            trajectory.append((action_name, node_pair, prob, value))
            graph, edges = apply_action(graph, edges, action_name, node_pair)
        graph, nodes, edges = self.revert_graph(graph, nodes, edges, trajectory)
        return trajectory, graph, nodes, edges

    def compute_real_values(self, trajectory, graph, nodes, edges, questions_answers):
        # walk through the trajectory and generate answers
        real_values = []
        for i, (action_name, node_pair, _, _) in enumerate(trajectory):
            if action_name == "stop":
                break
            graph, edges = apply_action(graph, edges, action_name, node_pair)
            results = rag(graph, nodes, edges, questions_answers)
            # compute the mean score of the generated answers
            real_value = sum(score for _, _, _, score in results) / len(results)
            real_values.append(real_value)

        # reset the graph to the initial state
        graph, nodes, edges = self.revert_graph(graph, nodes, edges, trajectory)

        return real_values, graph, nodes, edges

    def episode(self, graph, nodes, edges, questions_answers):
        # Set networks to evaluation mode for the episode
        self.policy_net.eval()
        self.critic_net.eval()
        self.add_net.eval()
        self.remove_net.eval()
        with torch.inference_mode():
            starting_value = compute_graph_hash(graph, nodes, edges)
            # Collect a trajectory. The trajectory will modify the graph and edges
            trajectory, graph, nodes, edges = self.collect_trajectory(graph, nodes, edges)
            self.check_graph_hash(graph, nodes, edges, starting_value)
            # Calculate the return
            advantages, returns = self.compute_advantages(trajectory)
            self.check_graph_hash(graph, nodes, edges, starting_value)
            # Compute the real values
            real_values, graph, nodes, edges = self.compute_real_values(
                trajectory, graph, nodes, edges, questions_answers
            )
            self.check_graph_hash(graph, nodes, edges, starting_value)

        # PPO update
        graph, nodes, edges = self.ppo_update(trajectory, advantages, returns, real_values, graph, nodes, edges)
        self.check_graph_hash(graph, nodes, edges, starting_value)

    def check_graph_hash(self, graph, nodes, edges, starting_value):
        ending_value = compute_graph_hash(graph, nodes, edges)
        assert starting_value == ending_value, "Graph hash mismatch"

    def revert_graph(self, graph, nodes, edges, trajectory):
        # Apply the trajectory in reverse to revert the graph to its initial state
        for action_name, node_pair, _, _ in reversed(trajectory):
            graph, edges = revert_action(graph, edges, action_name, node_pair)
        return graph, nodes, edges

    def compute_policy_loss(self, network, node1_emb, node2_emb, advantage):
        prob = network(node1_emb, node2_emb)
        target = (
            torch.tensor([1.0], dtype=torch.float32, device=node1_emb.device)
            if advantage > 0
            else torch.tensor([0.0], device=node1_emb.device, dtype=torch.float32)
        )
        target = target.to(prob.device)  # Ensure the target is on the same device as prob
        bce_loss = F.binary_cross_entropy(prob, target)
        return bce_loss * advantage.abs()

    def ppo_update(self, trajectory, advantages, returns, real_values, graph, nodes, edges, epsilon=0.2, epochs=EPOCHS):
        # move graph on the same device as the model
        graph = graph.to(next(self.policy_net.parameters()).device)
        # Set networks to training mode for the update phase
        self.policy_net.train()
        self.critic_net.train()
        self.add_net.train()
        self.remove_net.train()

        old_log_probs = torch.log(
            torch.tensor([prob for _, _, prob, _ in trajectory], dtype=torch.float32, device=graph.x.device)
        )

        advantages = torch.tensor(advantages, dtype=torch.float32, device=graph.x.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=graph.x.device)
        real_values = torch.tensor(real_values, dtype=torch.float32, device=graph.x.device)

        for _ in range(epochs):
            new_log_probs = []
            new_values = []
            action_losses = []

            # Recompute states and action probabilities by reapplying actions
            for i, (action_name, node_pair, prob, _) in enumerate(trajectory):
                action_probs = self.policy_net(graph.x, graph.edge_index)

                if action_name == "stop":
                    prob = action_probs[0, 2]  # Assuming the stop action is represented by the third probability
                    action_one_hot = torch.tensor([0, 0, 1], dtype=torch.float32, device=graph.x.device)
                    node1_emb = graph.x.mean(dim=0)  # Placeholder for stop action
                    node2_emb = graph.x.mean(dim=0)  # Placeholder for stop action
                elif action_name == "add":
                    prob = action_probs[0, 0]  # Use the first probability for add action
                    action_one_hot = torch.tensor([1, 0, 0], dtype=torch.float32, device=graph.x.device)
                    node1_emb = graph.x[node_pair[0]]
                    node2_emb = graph.x[node_pair[1]]
                    action_loss = self.compute_policy_loss(self.add_net, node1_emb, node2_emb, advantages[i])
                    action_losses.append(action_loss)
                elif action_name == "remove":
                    prob = action_probs[0, 1]  # Use the second probability for remove action
                    action_one_hot = torch.tensor([0, 1, 0], dtype=torch.float32, device=graph.x.device)
                    node1_emb = graph.x[node_pair[0]]
                    node2_emb = graph.x[node_pair[1]]
                    action_loss = self.compute_policy_loss(self.remove_net, node1_emb, node2_emb, advantages[i])
                    action_losses.append(action_loss)

                new_log_probs.append(torch.log(prob))

                action_prob_tensor = torch.tensor([prob.item()], dtype=torch.float32, device=graph.x.device).unsqueeze(
                    0
                )
                value = self.critic_net(
                    graph.x,
                    graph.edge_index,
                    action_one_hot.unsqueeze(0),
                    node1_emb.unsqueeze(0),
                    node2_emb.unsqueeze(0),
                    action_prob_tensor,
                )
                new_values.append(value)
                graph, edges = apply_action(graph, edges, action_name, node_pair)

            new_log_probs = torch.stack(new_log_probs)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Calculate the clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Combine all losses
            total_policy_loss = policy_loss + torch.stack(action_losses).mean()

            # Update policy network, action networks, and value network in one step
            new_values = torch.stack(new_values).squeeze()
            value_loss = self.value_loss_fn(new_values, real_values)

            total_loss = total_policy_loss + value_loss

            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.optimizer.step()

            print(
                f"Total Loss: {total_loss.item():.4f}, Total Policy Loss: {total_policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}"
            )

            # Revert the graph to its initial state at the end of each epoch
            graph, nodes, edges = self.revert_graph(graph, nodes, edges, trajectory)

        return graph, nodes, edges

    def infer_trajectory(self, graph, nodes, edges, min_steps=MIN_STEPS, max_steps=MAX_STEPS):
        trajectory = []
        self.policy_net.eval()
        self.add_net.eval()
        self.remove_net.eval()
        self.critic_net.eval()

        with torch.inference_mode():
            for i in range(max_steps):
                action_probs = self.policy_net(graph.x, graph.edge_index)
                action_name, node_pair, prob, value = sample_action(
                    graph, self.add_net, self.remove_net, self.critic_net, action_probs
                )
                if action_name == "stop":
                    if i >= min_steps:
                        break
                    else:
                        continue
                # Append the action to the trajectory
                trajectory.append((action_name, node_pair, prob, value))
                # Apply the action to modify the graph
                graph, edges = apply_action(graph, edges, action_name, node_pair)
        return trajectory, graph, nodes, edges
