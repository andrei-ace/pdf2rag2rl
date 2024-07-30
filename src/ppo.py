from actions import apply_action, revert_action, sample_action
from graphs import EMBEDDING_DIM, compute_graph_hash
from models import AddNetwork, CriticNetwork, PolicyNetwork, RemoveNetwork


class PPO:
    def __init__(self):

        input_dim = EMBEDDING_DIM
        hidden_dim = 64
        projection_dim = 32

        # Initialize networks
        self.add_net = AddNetwork(input_dim, hidden_dim)
        self.remove_net = RemoveNetwork(input_dim, hidden_dim)
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        self.critic_net = CriticNetwork(input_dim, hidden_dim, projection_dim)

    def compute_advantages(self, trajectory, graph, nodes, edges, gamma=0.99):
        advantages = []
        returns = []
        G = 0  # Initialize the expected return

        # Iterate over the trajectory in reverse
        for action_name, node_pair, prob, value in reversed(trajectory):
            # Calculate the return
            G = value + gamma * G
            returns.insert(0, G)

            # Calculate the advantage
            advantage = G - value
            advantages.insert(0, advantage)

            # Revert the action
            graph, edges = revert_action(graph, edges, action_name, node_pair)

        return advantages, returns, graph, nodes, edges

    def collect_trajectory(self, graph, nodes, edges, min_steps=50, max_steps=100):
        trajectory = []
        for i in range(max_steps):
            action_probs = self.policy_net(graph.x, graph.edge_index)
            action_name, node_pair, prob, value = sample_action(
                graph, self.add_net, self.remove_net, self.critic_net, action_probs
            )
            if action_name == "stop" and i >= min_steps:
                break
            # save the action, node pair, probability, and value
            trajectory.append((action_name, node_pair, prob, value))
            graph, edges = apply_action(graph, edges, action_name, node_pair)
        return trajectory, graph, nodes, edges

    def compute_real_values(self, trajectory, graph, nodes, edges):
        # return a list of 0s for now
        return [0] * len(trajectory), graph, nodes, edges

    def episode(self, graph, nodes, edges):

        starting_value = compute_graph_hash(graph, nodes, edges)
        # Collect a trajectory. The trajectory will modify the graph and edges
        trajectory, graph, nodes, edges = self.collect_trajectory(graph, nodes, edges)
        # Calculate the return
        advantages, returns, graph, nodes, edges = self.compute_advantages(trajectory, graph, nodes, edges)
        ending_value = compute_graph_hash(graph, nodes, edges)
        # Mental health check
        assert starting_value == ending_value, "Graph hash mismatch"

        # Compute the real values
        real_values, graph, nodes, edges = self.compute_real_values(trajectory, graph, nodes, edges)
        ending_value = compute_graph_hash(graph, nodes, edges)
        # Mental health check
        assert starting_value == ending_value, "Graph hash mismatch"

        # PPO update
        self.ppo_update(trajectory, advantages, returns, real_values, graph, nodes, edges)

    def ppo_update(self, trajectory, advantages, returns, real_values, graph, nodes, edges, epsilon=0.2):
        pass
