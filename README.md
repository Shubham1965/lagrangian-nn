
# Lagrangian Neural Network 

This Python script is part of a project designed to simulate and analyze the dynamics of a double pendulum system using lagrangian neural networks. The first half of the code deals with setting up the simulation environment, preparing datasets for training neural network models, and initializing data loaders for model training. Here's a detailed explanation of each part:

## Setting Up the Simulation:

```python
train = False
dp = DoublePendulum(g=9.81, m1=1, m2=1, L1=1, L2=1)
```

- `train` is a boolean variable used to toggle between training and testing modes of the neural network models.
- `dp` initializes an instance of the `DoublePendulum` class, which models the physics of a double pendulum system. The parameters passed to the constructor are the gravitational acceleration (`g`), masses of the pendulums (`m1`, `m2`), and lengths of the pendulum rods (`L1`, `L2`).

## Preparing the Dataset

```python
dp.get_training_data(60000)
```

- This line calls a method to generate training data for the neural networks. It simulates 60,000 random initial states of the double pendulum and computes the corresponding instantaneous accelerations and Lagrangian values and saves them.

## Loading the Dataset

```python
x_data = np.loadtxt('double_pendulum_x_train.txt', delimiter=',')
y_data_acc = np.loadtxt('double_pendulum_y_acc_train.txt', delimiter=',')
y_data_lg = np.loadtxt('double_pendulum_y_lagrangian_train.txt', delimiter=',')
```

- These lines load the pre-generated training data from text files. `x_data` contains the initial states, `y_data_acc` contains the instantaneous accelerations, and `y_data_lg` contains the Lagrangian values of the double pendulum system for each initial state.

## Creating Dataset Instances

```python
dataset_acc = DoublePendulumDataset(x_data, y_data_acc)
dataset_lg = DoublePendulumDataset(x_data, y_data_lg)
```

- Two instances of `DoublePendulumDataset` are created for training two different models: one for predicting instantaneous accelerations (`dataset_acc`) and another one for predicting Lagrangian (`dataset_lg`).

## Splitting the Dataset

```python
split = 0.8	
train_size = int(split * len(dataset_acc))
valid_size = len(dataset_acc) - train_size
```

- The dataset is split into training and validation sets based on an 80-20 ratio.

## Creating Data Loaders

```python
trainloader_acc = DataLoader(traindataset_acc, batch_size=32, shuffle=True, num_workers=0)
validloader_acc = DataLoader(validdataset_acc, batch_size=32, shuffle=True, num_workers=0)
trainloader_lg = DataLoader(traindataset_lg, batch_size=32, shuffle=True, num_workers=0)
validloader_lg = DataLoader(validdataset_lg, batch_size=32, shuffle=True, num_workers=0)
```

- Data loaders for both the acceleration-based and Lagrangian datasets are created for training and validation. These loaders automate the process of batching, shuffling, and loading the data during the training process.

## Setting the Device for Training

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- This line checks for the availability of a GPU with CUDA support for faster computation. If a GPU is not available, it falls back to using the CPU.

---

## Neural Network Training

When `train` is set to `True`, the code enters the training phase for two neural networks: a baseline network for predicting instantaneous accelerations and a Lagrangian neural network (LNN) that incorporates physical laws into its architecture.

### Baseline Neural Network

```python
net = Baseline(input_dim=4, layers=4, hidden_size=500, output_dim=2)
net = net.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
net.apply(weights_init_normal)
acc_losses = train_model_baseline(net, criterion, optimizer, trainloader_acc, validloader_acc, model_save='baseline_double_pendulum_acc.pt', epochs=400)
```

- A baseline neural network is defined with specific input dimensions, layers, hidden sizes, and output dimensions. It's moved to the available computing device (GPU or CPU).
- The L1 loss function is chosen for its ability to handle outliers, and the Adam optimizer is used for its efficiency in handling sparse gradients and adaptive learning rates.
- Weights are initialized using a normal distribution, aiming to improve convergence during training.

### Training Loop

The training loop involves feeding batches of data to the network, calculating the loss, and updating the model's parameters. This process is repeated for a specified number of epochs. Training and validation losses are plotted to monitor the model's learning progress.

### Lagrangian Neural Network (LNN)

```python
LN_net = lnn(input_dim=4, layers=4, hidden_size=500, output_dim=1)
LN_net = LN_net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(LN_net.parameters(), lr=1e-3)
LN_net.apply(weights_init_custom)
ln_losses = train_model_lnn(LN_net, criterion, optimizer, trainloader_lg, validloader_lg, model_save='lnn_double_pendulum.pt', epochs=400)
```

- A Lagrangian neural network is similarly initialized but with a focus on modeling the system using the principles of Lagrangian mechanics. This model aims to learn the underlying physical laws governing the double pendulum system.
- The mean squared error (MSE) loss function is chosen for its effectiveness in regression tasks, and custom weight initialization is applied to potentially enhance learning specific to physical systems.

## Model Testing and Simulation

When `train` is set to `False`, the code loads the trained models and uses them to simulate the double pendulum's motion. The simulation compares the trajectories predicted by the neural networks against the actual dynamics.

### Loading Trained Models

```python
# Baseline model
with open('baseline_double_pendulum_acc.pt', 'rb') as f:
    net.load_state_dict(load(f))

# Lagrangian neural network
with open('lnn_double_pendulum.pt', 'rb') as f:
    LN_net.load_state_dict(load(f))
```

- Trained models are loaded from saved state dictionaries, allowing for the evaluation of their performance on simulating the double pendulum dynamics.

### Simulating Trajectories

The code calculates the trajectory of the double pendulum over time using both the baseline and LNN models. It integrates the predicted accelerations to update the pendulum's state across discrete time steps.

### Visualization

```python
# Plotting trajectories
plt.subplots(...)
plt.plot(...)
```

- The trajectories, including angles and angular velocities of both pendulums, are plotted over time. This visualization allows for a direct comparison between the actual dynamics of the double pendulum and the predictions made by the trained neural networks.

