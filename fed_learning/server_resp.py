import time
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


class MyFlowerServer(fl.server.Server):
    def __init__(self):
        super().__init__(
            strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
            min_fit_clients=1,  #  Sample 3 clients for training
            min_evaluate_clients=1,
            min_available_clients=1,
            evaluate_metrics_aggregation_fn=self.weighted_average
            ),
        )
        self.response_times = []

    def on_fit_complete(self, result: fl.common.FitRes, **kwargs):
        # Calculate the response time in seconds
        response_time = time.time() - result.client_state["start_time"]
        self.response_times.append(response_time)
        print(f"Round {result.round_num} took {response_time:.4f} seconds.")
    
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    server = MyFlowerServer()
    server.start_server(config={"num_rounds": 10}, server_address="127.0.0.1:8080")
