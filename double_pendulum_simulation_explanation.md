
# Explaining the Double Pendulum Simulation Code

This Python script is part of a project designed to simulate and analyze the dynamics of a double pendulum system using neural networks. The first half of the code deals with setting up the simulation environment, preparing datasets for training neural network models, and initializing data loaders for model training. Here's a detailed explanation of each part:

## Setting Up the Simulation

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

- This line calls a method to generate training data for the neural networks. It simulates 60,000 random initial states of the double pendulum and computes the corresponding instantaneous accelerations and Lagrangian values.

## Loading the Dataset

```python
x_data = np.loadtxt('double_pendulum_x_train.txt', delimiter=',')
y_data_acc = np.loadtxt('double_pendulum_y_acc_train.txt', delimiter=',')
y_data_lg = np.loadtxt('double_pendulum_y_lagrangian_train.txt', delimiter=',')
```

- These lines load the pre-generated training data from text files. `x_data` contains the initial states, `y_data_acc` contains the instantaneous accelerations, and `y_data_lg` contains the Lagrangian values of the system for each initial state.

## Creating Dataset Instances

```python
dataset_acc = DoublePendulumDataset(x_data, y_data_acc)
dataset_lg = DoublePendulumDataset(x_data, y_data_lg)
```

- Two instances of `DoublePendulumDataset` are created for training two different models: one for predicting instantaneous accelerations (`dataset_acc`) and another for working with Lagrangian mechanics (`dataset_lg`).

## Splitting the Dataset

```python
split = 0.8	
train_size = int(split * len(dataset_acc))
valid_size = len(dataset_acc) - train_size
```

- The dataset is split into training and validation sets based on an 80-20 ratio. The size of each subset is calculated accordingly.

## Creating Data Loaders

```python
trainloader_acc = DataLoader(traindataset_acc, batch_size=32, shuffle=True, num_workers=0)
validloader_acc = DataLoader(validdataset_acc, batch_size=32, shuffle=True, num_workers=0)
trainloader_lg = DataLoader(traindataset_lg, batch_size=32, shuffle=True, num_workers=0)
validloader_lg = DataLoader(validdataset_lg, batch_size=32, shuffle=True, num_workers=0)
```

- Data loaders for both the acceleration-based and Lagrangian datasets are created for training and validation sets. These loaders automate the process of batching, shuffling, and loading the data during the training process.

## Setting the Device for Training

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- This line checks for the availability of a GPU with CUDA support for faster computation. If a GPU is not available, it falls back to using the CPU.

---

This explanation covers the setup and preliminary stages of the simulation code, focused on initializing the simulation environment, preparing the datasets, and setting up the data loaders for neural network training.
