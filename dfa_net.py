import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import math
import os
from tqdm import tqdm

dataset_path = "/home/mgabbita/nobackup/autodelete/fashion-mnist"

############# DATA ###########################

def load_idx_images(filename):
    with open(filename, 'rb') as f:
        # Read the 16-byte header using big-endian 32-bit integers ('>i4')
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.dtype('>i4'))
        
        # Read the rest of the file as unsigned bytes
        image_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        
        # Reshape to (Batch, Channels, Height, Width) for PyTorch
        images = image_data.reshape(num_images, 1, rows, cols)
        
        # Convert to tensor and normalize from [0, 255] to [-1.0, 1.0]
        tensor_images = torch.tensor(images, dtype=torch.float32)
        tensor_images = (tensor_images / 127.5) - 1.0
        
        return tensor_images

def load_idx_labels(filename):
    with open(filename, 'rb') as f:
        # Read the 8-byte header using big-endian 32-bit integers ('>i4')
        magic, num_labels = np.frombuffer(f.read(8), dtype=np.dtype('>i4'))
        
        # Read the rest of the file
        label_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        
        # Convert to a PyTorch long tensor (required for CrossEntropyLoss)
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
        # Save first batch of images to disk for verification`
        # Convert images from [-1.0, 1.0] back to [0, 255]
        images_to_save = ((images + 1.0) * 127.5).byte().numpy()
        # Save the first image in the batch
        from PIL import Image
        img = Image.fromarray(images_to_save[0, 0])  # Get the first image (1 channel, so we take the 0th index)
        img.save('sample_image.png')
        break  # Just check the first batch

############# MODEL ###########################

class DFALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, B_matrix, shared_state):
        # FIX 1: We must also save `weight` to correctly compute grad_input
        ctx.save_for_backward(input, weight, B_matrix)
        ctx.shared_state = shared_state 
        
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved tensors
        input, weight, B_matrix = ctx.saved_tensors
        shared_state = ctx.shared_state
        
        # Grab the global error [Batch, 10]
        global_error = shared_state['global_error']
        
        # FIX 2: Correctly project the error 
        # [64, 10] @ [10, 256] -> [64, 256]
        projected_error = global_error @ B_matrix.T 
        
        # FIX 3: Compute proper gradients for autograd
        # Pass gradient back to the layer below: [64, 256] @ [256, 784] -> [64, 784]
        grad_input = projected_error @ weight      
        
        # Gradient for this layer's weights: [256, 64] @ [64, 784] -> [256, 784]
        grad_weight = projected_error.T @ input    
        
        # Gradient for this layer's bias: [256]
        grad_bias = projected_error.sum(0) if ctx.needs_input_grad[2] else None
        
        return grad_input, grad_weight, grad_bias, None, None


class DFALinear(nn.Module):
    def __init__(self, in_features, out_features, error_dim, init_strategy, shared_state):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared_state = shared_state
        
        # Standard learnable weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Register B_matrix as a buffer so it saves with model state but DOES NOT update via autograd
        self.register_buffer('B_matrix', torch.Tensor(out_features, error_dim))
        
        self.init_weights(init_strategy)

    def init_weights(self, strategy):
        # Initialize standard weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Modular B_i initialization based on your experiment
        if strategy == 'random':
            nn.init.normal_(self.B_matrix)
        elif strategy == 'orthogonal':
            nn.init.orthogonal_(self.B_matrix)
        elif strategy == 'xavier':
            nn.init.xavier_normal_(self.B_matrix)

    def forward(self, input):
        # Pass everything into your custom autograd function
        return DFALinearFunction.apply(input, self.weight, self.bias, self.B_matrix, self.shared_state)
    
class DFANetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, init_strategy='orthogonal'):
        super().__init__()
        
        # 1. Create the bridge
        self.shared_state = {}
        
        # 2. Pass the bridge to every DFA layer
        self.layer1 = DFALinear(input_dim, hidden_dim, error_dim=output_dim, 
                                init_strategy=init_strategy, shared_state=self.shared_state)
        
        self.layer2 = DFALinear(hidden_dim, hidden_dim, error_dim=output_dim, 
                                init_strategy=init_strategy, shared_state=self.shared_state)
        
        # The final layer can be a standard linear layer because DFA only replaces 
        # backprop for the hidden layers. The final layer learns directly from the loss.
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.classifier(x)
        return x
    
def train_dfa_network(num_epochs, dataloader, do_tqdm=True):
    model = DFANetwork(input_dim=784, hidden_dim=256, output_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if do_tqdm:
        iteration = tqdm(range(num_epochs), desc="Training")
    else:
        iteration = range(num_epochs)

    # Training Loop
    for epochs in iteration:
        epoch_losses = []
        for x_batch, y_batch in dataloader:
            
            optimizer.zero_grad()
            
            # 1. Forward Pass
            predictions = model(x_batch)
            
            # 2. Compute standard loss for the final classifier layer
            loss = criterion(predictions, y_batch)
            
            # 3. INTERCEPT: Compute the global error and put it in the shared state
            # For cross-entropy, the error gradient with respect to logits is (softmax(preds) - one_hot(y))
            with torch.no_grad(): 
                probabilities = torch.softmax(predictions, dim=1)
                y_one_hot = torch.nn.functional.one_hot(y_batch, num_classes=10).float()
                global_error = probabilities - y_one_hot
                
                model.shared_state['global_error'] = global_error
                
            # 4. Backward Pass
            # The classifier uses standard autograd. 
            # The hidden layers ignore standard autograd and pull from shared_state.
            loss.backward()
            
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epochs+1}/{num_epochs}, Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")

if __name__ == "__main__":
    num_epochs = 64
    x_train = load_idx_images(os.path.join(dataset_path, 'train-images-idx3-ubyte'))
    y_train = load_idx_labels(os.path.join(dataset_path, 'train-labels-idx1-ubyte'))
    trainset = TensorDataset(x_train, y_train)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    train_dfa_network(num_epochs, trainloader)