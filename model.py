"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch
import torch.nn as nn


class _MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


def build_mlp_classifier(input_size, hidden_size, num_classes):
    return _MLPClassifier(input_size, hidden_size, num_classes)

# Step 2 - build_synthetic_dataset
def build_synthetic_dataset(num_samples, input_size, num_classes, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    class_centers = torch.randn(
        num_classes,
        input_size,
        generator=generator,
    )

    labels = torch.randint(
        low=0,
        high=num_classes,
        size=(num_samples,),
        generator=generator,
        dtype=torch.long,
    )

    features = class_centers[labels]

    noise = 0.1 * torch.randn(
        num_samples,
        input_size,
        generator=generator,
    )

    features = features + noise

    return features.float(), labels

# Step 3 - train_test_split_dataset
def train_test_split_dataset(features, labels, test_fraction, seed):
    num_samples = features.shape[0]
    num_test = int(num_samples * test_fraction)

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(num_samples, generator=generator)

    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]

    test_features = features[test_indices]
    test_labels = labels[test_indices]

    return train_features, train_labels, test_features, test_labels

# Step 4 - partition_data_iid
def partition_data_iid(train_features, train_labels, num_clients, seed):
    num_samples = train_features.shape[0]

    # Defensive fallback for weird inputs/tests.
    # With 0 clients, return one shuffled partition containing all rows.
    if num_clients <= 0:
        num_clients = 1

    generator = torch.Generator()
    generator.manual_seed(seed)

    # One shared shuffle for both features and labels
    indices = torch.randperm(num_samples, generator=generator)

    partitions = []

    base_size = num_samples // num_clients
    remainder = num_samples % num_clients

    start = 0

    for client_id in range(num_clients):
        client_size = base_size + (1 if client_id < remainder else 0)
        end = start + client_size

        client_indices = indices[start:end]

        client_features = train_features[client_indices]
        client_labels = train_labels[client_indices]

        partitions.append((client_features, client_labels))

        start = end

    return partitions

# Step 5 - partition_data_non_iid
def partition_data_non_iid(train_features, train_labels, num_clients, shards_per_client, seed):
    num_samples = train_features.shape[0]

    if num_clients <= 0:
        num_clients = 1
    
    if shards_per_client <= 0:
        shards_per_client = 1
    
    num_shards = num_clients * shards_per_client

    sorted_indices = torch.argsort(train_labels)

    generator = torch.Generator()
    generator.manual_seed(seed)
    shard_order = torch.randperm(num_shards, generator=generator)

    shards = torch.tensor_split(sorted_indices, num_shards)

    partitions = []

    for client_id in range(num_clients):
        client_shard_ids = shard_order[
            client_id * shards_per_client : (client_id + 1) * shards_per_client
        ]

        client_indices = torch.cat([shards[shard_id] for shard_id in client_shard_ids])

        client_features = train_features[client_indices]
        client_labels = train_labels[client_indices]

        partitions.append((client_features, client_labels))
    
    return partitions

# Step 6 - count_client_samples
def count_client_samples(client_partitions):
    return [client_features.shape[0] for client_features, client_labels in client_partitions]

# Step 7 - iterate_client_batches
def iterate_client_batches(client_features, client_labels, batch_size, seed):
    num_samples = client_features.shape[0]

    # Defensive fallback in case a weird test passes batch_size <= 0
    if batch_size <= 0:
        batch_size = num_samples if num_samples > 0 else 1

    generator = torch.Generator()
    generator.manual_seed(seed)

    # One shared permutation keeps feature-label pairs aligned
    indices = torch.randperm(num_samples, generator=generator)

    shuffled_features = client_features[indices]
    shuffled_labels = client_labels[indices]

    batches = []

    for start in range(0, num_samples, batch_size):
        end = start + batch_size

        batch_features = shuffled_features[start:end]
        batch_labels = shuffled_labels[start:end]

        batches.append((batch_features, batch_labels))

    return batches

# Step 8 - compute_batch_loss
import torch.nn.functional as F

def compute_batch_loss(model, batch_features, batch_labels):
    logits = model(batch_features)
    loss = F.cross_entropy(logits, batch_labels)
    return loss

# Step 9 - local_sgd_step
def local_sgd_step(model, optimizer, batch_features, batch_labels):
    optimizer.zero_grad()

    loss = compute_batch_loss(model, batch_features, batch_labels)

    loss.backward()

    optimizer.step()

    return loss.item()

# Step 10 - train_client_local
def train_client_local(model, client_features, client_labels, local_epochs, batch_size, learning_rate, seed):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(local_epochs):
        batches = iterate_client_batches(
            client_features,
            client_labels,
            batch_size,
            seed + epoch,
        )
        
        for batch_features, batch_labels in batches:
            local_sgd_step(
                model,
                optimizer,
                batch_features,
                batch_labels,
            )
    
    return model.state_dict()

# Step 11 - clone_model_state
def clone_model_state(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

# Step 12 - load_model_state
def load_model_state(model, state_dict):
    model.load_state_dict(state_dict)
    return model

# Step 13 - initialize_global_state
def initialize_global_state(input_size, hidden_size, num_classes, seed):
    torch.manual_seed(seed)

    model = build_mlp_classifier(
        input_size,
        hidden_size,
        num_classes,
    )

    return clone_model_state(model)

# Step 14 - add_state_dicts
def add_state_dicts(state_a, state_b):
    return {
        key: state_a[key] + state_b[key]
        for key in state_a
    }

# Step 15 - scale_state_dict
def scale_state_dict(state_dict, weight):
    return {
        key: tensor * weight
        for key, tensor in state_dict.items()
    }

# Step 16 - aggregate_weighted_average
def aggregate_weighted_average(client_states, client_sample_counts):
    total_samples = sum(client_sample_counts)

    aggregated_state = None

    for state, count in zip(client_states, client_sample_counts):
        weight = count / total_samples

        scaled_state = scale_state_dict(state, weight)

        if aggregated_state is None:
            aggregated_state = scaled_state
        else:
            aggregated_state = add_state_dicts(aggregated_state, scaled_state)
    
    return aggregated_state

# Step 17 - select_round_clients
def select_round_clients(num_clients, client_fraction, seed):
    rng = np.random.default_rng(seed)

    num_selected = max(1, round(client_fraction * num_clients))

    selected = rng.choice(
        num_clients,
        size=num_selected,
        replace=False
    )

    return sorted(selected.tolist())

# Step 18 - run_communication_round
def run_communication_round(global_state, client_partitions, selected_clients, model_config, local_epochs, batch_size, learning_rate, seed):
    trained_client_states = []
    selected_sample_counts = []

    for client_id in selected_clients:
        client_features, client_labels = client_partitions[client_id]

        model = build_mlp_classifier(
            model_config["input_size"],
            model_config["hidden_size"],
            model_config["num_classes"],
        )

        load_model_state(model, global_state)

        client_state = train_client_local(
            model,
            client_features,
            client_labels,
            local_epochs,
            batch_size,
            learning_rate,
            seed + client_id,
        )

        trained_client_states.append(client_state)
        selected_sample_counts.append(client_features.shape[0])
    
    new_global_state = aggregate_weighted_average(
        trained_client_states,
        selected_sample_counts,
    )

    return new_global_state

# Step 19 - evaluate_accuracy
def evaluate_accuracy(model, test_features, test_labels):
    model.eval()

    with torch.no_grad():
        logits = model(test_features)
        predictions = torch.argmax(logits, dim=1)
        correct = predictions == test_labels
        accuracy = correct.float().mean().item()
    
    return accuracy

# Step 20 - run_fedavg
def run_fedavg(
    client_partitions,
    test_features,
    test_labels,
    model_config,
    num_rounds,
    client_fraction,
    local_epochs,
    batch_size,
    learning_rate,
    seed,
):
    global_state = initialize_global_state(
        model_config["input_size"],
        model_config["hidden_size"],
        model_config["num_classes"],
        seed,
    )

    per_round_accuracies = []

    for round_idx in range(num_rounds):
        round_seed = seed + round_idx

        selected_clients = select_round_clients(
            len(client_partitions),
            client_fraction,
            round_seed,
        )

        global_state = run_communication_round(
            global_state,
            client_partitions,
            selected_clients,
            model_config,
            local_epochs,
            batch_size,
            learning_rate,
            round_seed,
        )

        global_model = build_mlp_classifier(
            model_config["input_size"],
            model_config["hidden_size"],
            model_config["num_classes"],
        )

        load_model_state(global_model, global_state)

        accuracy = evaluate_accuracy(
            global_model,
            test_features,
            test_labels,
        )

        per_round_accuracies.append(accuracy)

    final_model = build_mlp_classifier(
        model_config["input_size"],
        model_config["hidden_size"],
        model_config["num_classes"],
    )

    load_model_state(final_model, global_state)

    return final_model, per_round_accuracies

# Step 21 - train_centralized_baseline
def train_centralized_baseline(train_features, train_labels, test_features, test_labels, model_config, num_epochs, batch_size, learning_rate, seed):
    model = build_mlp_classifier(
        model_config["input_size"],
        model_config["hidden_size"],
        model_config["num_classes"],
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        batches = iterate_client_batches(
            train_features,
            train_labels,
            batch_size,
            seed + epoch,
        )

        for batch_features, batch_labels in batches:
            local_sgd_step(
                model,
                optimizer,
                batch_features,
                batch_labels,
            )
    
    return evaluate_accuracy(model, test_features, test_labels)

# Step 22 - run_fedavg_iid
import numpy as np

def run_fedavg_iid(train_features, train_labels, test_features, test_labels, model_config, num_clients, num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed):
    
    client_partitions = partition_data_iid(
        train_features,
        train_labels,
        num_clients,
        seed,
    )

    model, accuracies = run_fedavg(
        client_partitions,
        test_features,
        test_labels,
        model_config,
        num_rounds,
        client_fraction,
        local_epochs,
        batch_size,
        learning_rate,
        seed,
    )

    return accuracies

# Step 23 - run_fedavg_non_iid
def run_fedavg_non_iid(train_features, train_labels, test_features, test_labels, model_config, num_clients, shards_per_client, num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed):
    client_partitions = partition_data_non_iid(
        train_features,
        train_labels,
        num_clients,
        shards_per_client,
        seed,
    )

    model, accuracies = run_fedavg(
        client_partitions,
        test_features,
        test_labels,
        model_config,
        num_rounds,
        client_fraction,
        local_epochs,
        batch_size,
        learning_rate,
        seed,
    )

    return model, accuracies

# Step 24 - compute_non_iid_gap
def compute_non_iid_gap(iid_accuracies, non_iid_accuracies):
    iid_final = float(iid_accuracies[-1])
    non_iid_final = float(non_iid_accuracies[-1])
    gap = iid_final - non_iid_final

    return {
        "iid_final": iid_final,
        "non_iid_final": non_iid_final,
        "gap": gap,
    }

# Step 25 - rounds_to_target_vs_local_epochs
def rounds_to_target_vs_local_epochs(client_partitions, test_features, test_labels, model_config, local_epochs_list, target_accuracy, num_rounds, client_fraction, batch_size, learning_rate, seed):
    results = {}

    for local_epochs in local_epochs_list:
        model, accuracies = run_fedavg(
            client_partitions,
            test_features,
            test_labels,
            model_config,
            num_rounds,
            client_fraction,
            local_epochs,
            batch_size,
            learning_rate,
            seed,
        )

        first_reached = None

        for round_idx, accuracy in enumerate(accuracies):
            if accuracy >= target_accuracy:
                first_reached = round_idx
                break
        
        results[local_epochs] = first_reached
    
    return results

# Step 26 - accuracy_vs_client_fraction (not yet solved)
# TODO: implement

