
# Second Half Explanation of the Double Pendulum Simulation Code

The second half of the code toggles between training neural network models and testing the trained models based on the `train` boolean flag. Here, we delve into the model training process, loss function selection, optimizer setup, and the testing phase where the models' performance is evaluated through simulation.

## Neural Network Training

When `train` is set to `True`, the code enters the training phase for two neural networks: a baseline network for predicting instantaneous accelerations and a Lagrangian neural network (LNN) that incorporates physical laws into its architecture.

### Baseline Neural Network

```python
net = Baseline(input_dim=4, layers=4, hidden_size=500, output_dim=2)
net = net.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
net.apply(weights_init_normal)
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

## Conclusion

This section of the code demonstrates the application of neural networks to model complex physical systems. By training models on simulated data and then testing their ability to predict the dynamics of a double pendulum, the code bridges the gap between theoretical physics and practical machine learning applications.
