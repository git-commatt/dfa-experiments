import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import math
import os
from tqdm import tqdm
import pickle
import yaml
import argparse

# Running this file with a valid config will start training.
# Don't run this file if you don't intend to run the training loop.

def load_config(config_file):
    """Load configuration from YAML file and set global variables."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    global dataset_path, experiment_name, output_dir, device
    dataset_path = config.get('dataset_path', "")
    experiment_name = config.get('experiment_name', "default_experiment")
    output_dir = config.get('output_dir', "output/experiment_results")
    device = config.get('device', "cuda" if torch.cuda.is_available() else "cpu")

############# DATA ###########################

def load_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.dtype('>i4'))
        image_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        images = image_data.reshape(num_images, 1, rows, cols)
        tensor_images = torch.tensor(images, dtype=torch.float32)
        tensor_images = (tensor_images / 127.5) - 1.0
        return tensor_images

def load_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype=np.dtype('>i4'))
        label_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        return torch.tensor(label_data, dtype=torch.long)

def test_data_loading():
    x_train = load_idx_images(os.path.join(dataset_path, 'train-images-idx3-ubyte'))
    y_train = load_idx_labels(os.path.join(dataset_path, 'train-labels-idx1-ubyte'))

    x_test = load_idx_images(os.path.join(dataset_path, 't10k-images-idx3-ubyte'))
    y_test = load_idx_labels(os.path.join(dataset_path, 't10k-labels-idx1-ubyte'))

    trainset = TensorDataset(x_train, y_train)
    testset = TensorDataset(x_test, y_test)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    for images, labels in trainloader:
        print("Batch of images shape:", images.shape)  # Should be (64, 1, 28, 28)
        print("Batch of labels shape:", labels.shape)  # Should be (64,)
        images_to_save = ((images + 1.0) * 127.5).byte().numpy()
        # Save the first image in the batch to verify it looks correct
        from PIL import Image
        img = Image.fromarray(images_to_save[0, 0])
        img.save('sample_image.png')
        break  # Just check the first batch

############# MODEL ###########################

class DFALinearFunction(torch.autograd.Function):
    # Custom autograd function to implement DFA logic in the backward pass
    @staticmethod
    def forward(ctx, input, weight, bias, B_matrix, shared_state):
        output = torch.nn.functional.linear(input, weight, bias)
        ctx.save_for_backward(input, weight, B_matrix, output)
        ctx.shared_state = shared_state
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, B_matrix, output = ctx.saved_tensors
        shared_state = ctx.shared_state
        
        global_error = shared_state['global_error']
        
        raw_projected_error = global_error @ B_matrix.T 
        
        relu_mask = (output > 0).float()
        projected_error = raw_projected_error * relu_mask

        grad_input = projected_error @ weight      
        grad_weight = projected_error.T @ input    
        grad_bias = projected_error.sum(0) if ctx.needs_input_grad[2] else None
        
        return grad_input, grad_weight, grad_bias, None, None


class DFALinear(nn.Module):
    # Version of nn.Linear that uses DFA in the backward pass
    def __init__(self, in_features, out_features, provided_B_matrix, shared_state):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared_state = shared_state
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) 
        
        self.register_buffer('B_matrix', provided_B_matrix)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return DFALinearFunction.apply(input, self.weight, self.bias, self.B_matrix, self.shared_state)
    
class DFANetwork(nn.Module):
    # Simple feedforward network with DFA linear layers and a final standard linear classifier
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=5, init_strategy='random'):
        super().__init__()
        self.shared_state = {}
        self.layers = nn.ModuleList()
        
        B_matrices = self._generate_B_matrices(hidden_dim, output_dim, num_hidden_layers, init_strategy)
        
        current_dim = input_dim
        for i in range(num_hidden_layers):
            layer = DFALinear(
                in_features=current_dim, 
                out_features=hidden_dim, 
                provided_B_matrix=B_matrices[i], 
                shared_state=self.shared_state
            )
            self.layers.append(layer)
            current_dim = hidden_dim
            
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def _generate_B_matrices(self, hidden_dim, output_dim, num_layers, strategy):
        # Generate the B matrices for the network using the specified initialization strategy
        matrices = []
        
        if strategy == 'mutually_orthogonal':
            total_columns_needed = output_dim * num_layers
            
            if total_columns_needed > hidden_dim:
                raise ValueError(f"Hidden dim {hidden_dim} is too small to support {num_layers} mutually exclusive {output_dim}D spaces.")
                
            global_space = torch.empty(hidden_dim, hidden_dim)
            nn.init.orthogonal_(global_space)
            
            for i in range(num_layers):
                start_col = i * output_dim
                end_col = start_col + output_dim
                B_slice = global_space[:, start_col:end_col].clone()
                matrices.append(B_slice)
                
        elif strategy == 'independant_orthogonal':
            for _ in range(num_layers):
                B = torch.empty(hidden_dim, output_dim)
                nn.init.orthogonal_(B)
                matrices.append(B)
                
        elif strategy == 'random':
            for _ in range(num_layers):
                B = torch.empty(hidden_dim, output_dim)
                
                std_dev = 1.0 / math.sqrt(hidden_dim) 
                nn.init.normal_(B, mean=0.0, std=std_dev)
                
                matrices.append(B)
        elif strategy == 'random_identical':
            B = torch.empty(hidden_dim, output_dim)
            std_dev = 1.0 / math.sqrt(hidden_dim) 
            nn.init.normal_(B, mean=0.0, std=std_dev)
            for _ in range(num_layers):
                matrices.append(B.clone())

        elif strategy == 'zeros':
            for _ in range(num_layers):
                B = torch.zeros(hidden_dim, output_dim)
                matrices.append(B)
                
        return matrices

    def forward(self, x):

        x = torch.flatten(x, 1)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.classifier(x)
    
def train_dfa_network(num_epochs, dataloader, do_tqdm=True, init_strategy='mutually_orthogonal'):
    # Train a single DFA network
    model = DFANetwork(input_dim=784, hidden_dim=54, output_dim=10, num_hidden_layers=5, init_strategy=init_strategy)
    model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()


    if do_tqdm:
        iteration = tqdm(range(num_epochs), desc="Training")
    else:
        iteration = range(num_epochs)

    epoch_avg_losses = []

    # Training Loop
    for epochs in iteration:
        epoch_losses = []
        for x_batch, y_batch in dataloader:

            optimizer.zero_grad()
            
            predictions = model(x_batch)
            
            loss = criterion(predictions, y_batch)
            with torch.no_grad(): 
                probabilities = torch.softmax(predictions, dim=1)
                y_one_hot = torch.nn.functional.one_hot(y_batch, num_classes=10).float()
                global_error = (probabilities - y_one_hot) / x_batch.size(0)
                
                model.shared_state['global_error'] = global_error
                
            loss.backward()
            
            optimizer.step()
            epoch_losses.append(loss.item())
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_avg_losses.append(epoch_avg_loss)
    return epoch_avg_losses

if __name__ == "__main__":

    # Parse path to config file from command line arguments
    parser = argparse.ArgumentParser(description='Train DFA Networks with different initialization strategies.')
    parser.add_argument('--config', type=str, default='configs/example_config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Set up experiment parameters
    load_config(args.config) # Parameters are loaded from config to global variables
    init_strategies = ['zeros', 'random_identical', 'random', 'mutually_orthogonal']
    num_epochs = 32
    runs_per_strategy = 5

    # Set up dataloaders
    x_train = load_idx_images(os.path.join(dataset_path, 'train-images-idx3-ubyte')).to(device)
    y_train = load_idx_labels(os.path.join(dataset_path, 'train-labels-idx1-ubyte')).to(device)
    x_val = load_idx_images(os.path.join(dataset_path, 't10k-images-idx3-ubyte')).to(device)
    y_val = load_idx_labels(os.path.join(dataset_path, 't10k-labels-idx1-ubyte')).to(device)
    trainset = TensorDataset(x_train, y_train)
    valset = TensorDataset(x_val, y_val)
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
    valloader = DataLoader(valset, batch_size=1024, shuffle=False)

    # Run all experiments
    for strategy in init_strategies:
        strategy_results = {}
        for run in range(runs_per_strategy):
            
            strategy_results[(strategy, run)] = train_dfa_network(num_epochs, trainloader, init_strategy=strategy)
        out_name = f"{experiment_name}_{strategy}_results.pkl"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pickle.dump(strategy_results, open(os.path.join(output_dir, out_name), "wb"))