import numpy as np

from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import save, load

import matplotlib.pyplot as plt
from matplotlib.patches import Circle



class Integrate:

    def __init__(self, integrator, dt, t):
        self.integrator = integrator
        self.dt = dt
        self.t = t

    def euler_step(self, f, state):
        """
        Performs one step of Euler integration.
        """
        return state + self.dt * f(state)
    
    def rk4_step(self, f, state):
        """
        Performs one step of Runge-Kutta integration.
        """
        k1 = f(state)
        k2 = f(state + 0.5*self.dt*k1)
        k3 = f(state + 0.5*self.dt*k2)
        k4 = f(state + self.dt*k3)

        return state + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    def get_trajectory(self,f, state):
        """
        Integrates the system from t=0 to t=self.t
        """

        N = int(self.t/self.dt)
        trajectory = np.zeros((N, len(state)))

        for i in range(N):

            if self.integrator == 'euler':
                state = self.euler_step(f, state)
            else:
                state = self.rk4_step(f, state)
            
            trajectory[i, :] = state
        
        return trajectory
    
class DoublePendulum:

    def __init__(self, g, m1, m2, L1, L2):
        """
        Constructs a double pendulum simulator based on its
        Euler-Lagrange equations. Bob #1 is the one attached to the
        fixed pivot.

        g - The gravitational acceleration.
        m1 - The mass of bob #1.
        m2 - The mass of bob #2.
        L1 - The length of the rod for bob #1.
        L2 - The length of the rod for bob #2.
        """
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.n_states = 4

    def get_random_initial(self):
        y0 = np.random.rand(self.n_states) * 2 * np.pi - np.pi
        # y0[2], y0[3] = 0, 0 #if you do not want to have q_dot
        return y0

    def get_cartesian(self, trajectory):
        
        x1 = self.L1 * np.sin(trajectory[:, 0])
        y1 = -self.L1 * np.cos(trajectory[:, 0])
        x2 = x1 + self.L2 * np.sin(trajectory[:, 1])
        y2 = y1 - self.L2 * np.cos(trajectory[:, 1])
        return x1, y1, x2, y2

    def get_dynamics(self, state):
        
        # Unpack state
        theta1, theta2, theta1_dot, theta2_dot = state

        # Compute the derivatives
        dtheta1_dt = theta1_dot
        dtheta2_dt = theta2_dot
        ddtheta1_dt = (-self.g*(2*self.m1 + self.m2)*np.sin(theta1) - self.m2*self.g*np.sin(theta1 - 2*theta2) - 2*np.sin(theta1 - theta2)*self.m2*(theta2_dot**2*self.L2 + theta1_dot**2*self.L1*np.cos(theta1 - theta2)))/(self.L1*(2*self.m1 + self.m2 - self.m2*np.cos(2*theta1 - 2*theta2)))
        ddtheta2_dt = (2*np.sin(theta1 - theta2)*(theta1_dot**2*self.L1*(self.m1+self.m2) + self.g*(self.m1+self.m2)*np.cos(theta1) + theta2_dot**2*self.L2*self.m2*np.cos(theta1 - theta2)))/(self.L2*(2*self.m1 + self.m2 - self.m2*np.cos(2*theta1 - 2*theta2)))

        # Return the derivatives
        return np.array([dtheta1_dt, dtheta2_dt, ddtheta1_dt, ddtheta2_dt])
    
    def get_instantaneous_acceleration(self, state):
        
        # Unpack state
        theta1, theta2, theta1_dot, theta2_dot = state

        # Compute the accelerations
        ddtheta1_dt = (-self.g*(2*self.m1 + self.m2)*np.sin(theta1) - self.m2*self.g*np.sin(theta1 - 2*theta2) - 2*np.sin(theta1 - theta2)*self.m2*(theta2_dot**2*self.L2 + theta1_dot**2*self.L1*np.cos(theta1 - theta2)))/(self.L1*(2*self.m1 + self.m2 - self.m2*np.cos(2*theta1 - 2*theta2)))
        ddtheta2_dt = (2*np.sin(theta1 - theta2)*(theta1_dot**2*self.L1*(self.m1+self.m2) + self.g*(self.m1+self.m2)*np.cos(theta1) + theta2_dot**2*self.L2*self.m2*np.cos(theta1 - theta2)))/(self.L1*(2*self.m1 + self.m2 - self.m2*np.cos(2*theta1 - 2*theta2)))

        # Return the derivatives
        return np.array([ddtheta1_dt, ddtheta2_dt])

    # https://www.jousefmurad.com/engineering/double-pendulum-1/
    def get_lagrangian(self, state):
        # Unpack state
        theta1, theta2, theta1_dot, theta2_dot = state

        # Kinetic energy
        T = 0.5 * (self.m1 * (self.L1*theta1_dot)**2 +
                   self.m2 * ((self.L1*theta1_dot)**2 +
                               (self.L2*theta2_dot)**2 +
                               2*self.L1*self.L2*theta1_dot*theta2_dot*np.cos(theta1 - theta2)))

        # Potential energy
        V = -self.m1*self.g*self.L1*np.cos(theta1) - self.m2*self.g*(self.L1*np.cos(theta1) + self.L2*np.cos(theta2))

        # Return Lagrangian
        return T - V

    def get_hamiltonian(self, state):
        # Unpack state
        theta1, theta2, theta1_dot, theta2_dot = state

        # Kinetic energy
        T = 0.5 * (self.m1 * (self.L1*theta1_dot)**2 +
                   self.m2 * ((self.L1*theta1_dot)**2 +
                               (self.L2*theta2_dot)**2 +
                               2*self.L1*self.L2*theta1_dot*theta2_dot*np.cos(theta1 - theta2)))

        # Potential energy
        V = -self.m1*self.g*self.L1*np.cos(theta1) - self.m2*self.g*(self.L1*np.cos(theta1) + self.L2*np.cos(theta2))

        # Return Hamiltonian
        return T + V
    
    def get_training_data(self, samples):

        x_train, y_train_acc, y_train_lg = [], [], []

        for i in range(samples):

            state = self.get_random_initial()
            dstate = self.get_instantaneous_acceleration(state)
            lagrangian = self.get_lagrangian(state)

            x_train.append(state)
            y_train_acc.append(dstate)
            y_train_lg.append(lagrangian)

        np.savetxt('double_pendulum_x_train.txt', x_train, delimiter=',')
        np.savetxt('double_pendulum_y_acc_train.txt', y_train_acc, delimiter=',')
        np.savetxt('double_pendulum_y_lagrangian_train.txt', y_train_lg, delimiter=',')

    
    def plot_dblpend(self, ax, i, cart_coords, l1, l2, max_trail=30, trail_segments=20, r = 0.05):
        # Plot and save an image of the double pendulum configuration for time step i.
        plt.cla()

        x1, y1, x2, y2 = cart_coords
        ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k') # rods
        c0 = Circle((0, 0), r/2, fc='k', zorder=10) # anchor point
        c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10) # mass 1
        c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10) # mass 2
        ax.add_patch(c0)
        ax.add_patch(c1)
        ax.add_patch(c2)

        # plot the pendulum trail (ns = number of segments)
        s = max_trail // trail_segments
        for j in range(trail_segments):
            imin = i - (trail_segments-j)*s
            if imin < 0: continue
            imax = imin + s + 1
            alpha = (j/trail_segments)**2 # fade the trail into alpha
            ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                    lw=2, alpha=alpha)

        # Center the image on the fixed anchor point. Make axes equal.
        ax.set_xlim(-l1-l2-r, l1+l2+r)
        ax.set_ylim(-l1-l2-r, l1+l2+r)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')


    def fig2image(self, fig):
        
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image


    def get_dblpend_images(self, y, fig, ax, l1=1, l2=1, verbose=True):
        
        cart_coords = self.get_cartesian(y)
        images = [] ; di = 1
        N = len(y)
        for i in range(0, N, di):
            if verbose:
                print("{}/{}".format(i // di, N // di), end='\n' if i//di%15==0 else ' ')
            self.plot_dblpend(ax, i, cart_coords, l1, l2)
            images.append(self.fig2image(fig) )
        return images

class DoublePendulumDataset(Dataset):
    
    def __init__(self, x_data, y_data):
        
        self.x_data = x_data
        self.y_data = y_data
        self.n_samples = len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        sample = {'x': x, 'y': y}
        return sample

    def __len__(self):
        return self.n_samples

class Baseline(nn.Module):

    def __init__(self, input_dim, layers, hidden_size, output_dim):
        
        super(Baseline, self).__init__()

        self.input_dim = input_dim
        self.layers = layers
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_size)
        for i in range(layers):
            setattr(self, f'fc{i+2}', nn.Linear(hidden_size, hidden_size))
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        for i in range(self.layers):
            x = getattr(self, f'fc{i+2}')(x)
            x = F.relu(x)
        x = self.fc_out(x)
        return x
    
class lnn(nn.Module):

    def __init__(self, input_dim, layers, hidden_size, output_dim):

        super(lnn, self).__init__()
        self.input_dim = input_dim
        self.layers = layers
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_size)
        for i in range(layers):
            setattr(self, f'fc{i+2}', nn.Linear(hidden_size, hidden_size))
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = F.softplus(self.fc1(x))
        for i in range(self.layers):
            x = getattr(self, f'fc{i+2}')(x)
            x = F.softplus(x)
        x = self.fc_out(x)
        return x

    def get_acc(self,x): 

        grad = torch.autograd.functional.jacobian(self.forward, x, create_graph=True).reshape(1,self.input_dim)
        hess = torch.autograd.functional.hessian(self.forward, x, create_graph=True).reshape(self.input_dim,self.input_dim)

        nabla_qL = grad[0,0:2]
        hess_q_dotL = hess[2:4,2:4]
        hess_q_q_dotL = hess[2:4,0:2]
        lnn_net_acc = torch.linalg.pinv(hess_q_dotL) @ (nabla_qL - hess_q_q_dotL @ x[2:4])  

        return lnn_net_acc                                   
                                                                                                

def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)

def weights_init_custom(m):
    '''Initializes weights according to a custom rule for different types of layers.'''

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'is_first_layer') and m.is_first_layer:
            # Initialize weights for the first layer
            m.weight.data.normal_(0.0, 1)
            m.bias.data.fill_(0)
        elif hasattr(m, 'is_output_layer') and m.is_output_layer:
            # Initialize weights for the output layer
            m.weight.data.normal_(0.0, np.sqrt(1 / m.in_features))
            m.bias.data.fill_(0)
        else:
            # Initialize weights for hidden layers
            m.weight.data.normal_(0.0, np.sqrt(2 / (m.in_features + m.out_features)))
            m.bias.data.fill_(0)

def train_model_baseline(model, criterion, optimizer, trainloader, validloader, model_save, epochs):
    
    losses = np.empty((epochs, 2))

    # Train the network
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            running_loss = 0.0

            for _, data in enumerate(trainloader, 0):
                inputs, labels = data['x'].to(device).to(torch.float32), data['y'].to(device).to(torch.float32)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss/len(trainloader)

            model.eval() 
            validation_loss = 0.0

            with torch.no_grad():
                for _, data in enumerate(validloader, 0):
                    inputs, labels = data['x'].to(device).to(torch.float32), data['y'].to(device).to(torch.float32)
                    outputs = model(inputs)
                    valid_loss = criterion(outputs, labels)
                    validation_loss += valid_loss.item()

            valid_loss = validation_loss/len(validloader)
            losses[epoch] = [train_loss, valid_loss]
            pbar.set_description(f"Loss {train_loss:.02f}/{valid_loss:.02f}")

    with open(model_save, 'wb') as f:
        save(model.state_dict(), f)
    
    return losses

def train_model_lnn(model, criterion, optimizer, trainloader, validloader, model_save, epochs):
    
    losses = np.empty((epochs, 2))

    # Train the network
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            running_loss = 0.0

            for _, data in enumerate(trainloader, 0):
                inputs, labels = data['x'].to(device).to(torch.float32), data['y'].to(device).to(torch.float32)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss/len(trainloader)

            model.eval() 
            validation_loss = 0.0

            with torch.no_grad():
                for _, data in enumerate(validloader, 0):
                    inputs, labels = data['x'].to(device).to(torch.float32), data['y'].to(device).to(torch.float32)
                    outputs = model(inputs).squeeze(1)
                    valid_loss = criterion(outputs, labels)
                    validation_loss += valid_loss.item()

            valid_loss = validation_loss/len(validloader)
            losses[epoch] = [train_loss, valid_loss]
            pbar.set_description(f"Loss {train_loss:.02f}/{valid_loss:.02f}")

    with open(model_save, 'wb') as f:
        save(model.state_dict(), f)
    
    return losses

if __name__ == "__main__":
    
    train = False

    dp = DoublePendulum(g=9.81, m1=1, m2=1, L1=1, L2=1)
    
    # Create a dataset to store a random initial state and the corresponding instantaneous acceleration
    # dp.get_training_data(60000)

    # Load the dataset
    x_data = np.loadtxt('double_pendulum_x_train.txt', delimiter=',')
    y_data_acc = np.loadtxt('double_pendulum_y_acc_train.txt', delimiter=',')
    y_data_lg = np.loadtxt('double_pendulum_y_lagrangian_train.txt', delimiter=',')

    dataset_acc = DoublePendulumDataset(x_data, y_data_acc)
    # dataset_lg = DoublePendulumDataset(x_data, y_data_acc) # tried this but compute time very high, need to think about batch jacobians and hessians, and compute faster
    dataset_lg = DoublePendulumDataset(x_data, y_data_lg) # tried using this but does't work as a whole
    
    # Split the dataset into training and validation
    split = 0.8	
    train_size = int(split * len(dataset_acc))
    valid_size = len(dataset_acc) - train_size

    traindataset_acc, validdataset_acc = torch.utils.data.random_split(dataset_acc, [train_size, valid_size])
    traindataset_lg, validdataset_lg = torch.utils.data.random_split(dataset_lg, [train_size, valid_size])

    # Create dataloaders
    trainloader_acc = DataLoader(traindataset_acc, batch_size=32, shuffle=True, num_workers=0)
    validloader_acc = DataLoader(validdataset_acc, batch_size=32, shuffle=True, num_workers=0)
    trainloader_lg = DataLoader(traindataset_lg, batch_size=32, shuffle=True, num_workers=0)
    validloader_lg = DataLoader(validdataset_lg, batch_size=32, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train:
        # # Create the neural network
        # net = Baseline(input_dim=4, layers=4, hidden_size=500, output_dim=2)
        # net = net.to(device)
        # criterion = nn.L1Loss()
        # optimizer = optim.Adam(net.parameters(), lr=1e-3)

        # # Initialize the weights:
        # net.apply(weights_init_normal)

        # acc_losses = train_model_baseline(net, criterion, optimizer, trainloader_acc, validloader_acc, model_save='baseline_double_pendulum_acc.pt', epochs=400)

        # # Plot the training and validation losses
        # acc_losses = np.array(acc_losses)
        # plt.plot(np.arange(len(acc_losses)), acc_losses[:,0], label="train")
        # plt.plot(np.arange(len(acc_losses)), acc_losses[:,1], label="validation")
        # plt.legend()
        # plt.xlabel("epoch")
        # plt.ylabel("L1 Loss of Instantaneous Acceleration")
        # plt.tight_layout()
        # plt.savefig('baseline_double_pendulum_acc.png')

        # Create the neural network
        LN_net = lnn(input_dim=4, layers=4, hidden_size=500, output_dim=1)
        LN_net = LN_net.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(LN_net.parameters(), lr=1e-3)

        LN_net.apply(weights_init_custom)

        ln_losses = train_model_lnn(LN_net, criterion, optimizer, trainloader_lg, validloader_lg, model_save='lnn_double_pendulum.pt', epochs=400)

        ln_losses = np.array(ln_losses)
        plt.plot(np.arange(len(ln_losses)), ln_losses[:,0], label="train")
        plt.plot(np.arange(len(ln_losses)), ln_losses[:,1], label="validation")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("MSE Loss of Lagrangian")
        plt.tight_layout()
        plt.savefig('lnn_double_pendulum.png')
        plt.show()

    else: #test the trained models

        # To get simulation trajectories for the learned network later for comparision.
        intODE = Integrate('euler', 0.01, 10)
        trajectory = intODE.get_trajectory(dp.get_dynamics, [np.pi/2, np.pi/2, 0, 0])
        t = np.arange(0, intODE.t, intODE.dt)


        # Load the baseline model
        net = Baseline(input_dim=4, layers=4, hidden_size=500, output_dim=2)
        with open('baseline_double_pendulum_acc.pt', 'rb') as f:
            net.load_state_dict(load(f))

        trajectory_net = np.zeros((trajectory.shape[0], 4))
        trajectory_net[0,:] = trajectory[0,:]
        for i in range(1,len(trajectory)):
            net_acc = net(torch.tensor(trajectory[i-1,:]).float()).detach().numpy()
            trajectory_net[i,:] = trajectory[i-1,:] + intODE.dt*np.array([trajectory_net[i-1,0], trajectory_net[i-1,1], net_acc[0], net_acc[1]])

        # Load the lnn model
        LN_net = lnn(input_dim=4, layers=4, hidden_size=500, output_dim=1)
        with open('lnn_double_pendulum.pt', 'rb') as f:
            LN_net.load_state_dict(load(f))
        

        trajectory_lnn = np.zeros((trajectory.shape[0], 4))
        trajectory_lnn[0,:] = trajectory[0,:]
        for i in range(1,len(trajectory)):
            # print(LN_net.get_parameter(torch.tensor(trajectory[i-1,:], requires_grad=True, dtype=torch.float)))
            input = torch.tensor(trajectory_lnn[i-1,:], requires_grad=True, dtype=torch.float)
            lnn_net_acc = LN_net.get_acc(input).detach().numpy()
            trajectory_lnn[i,:] = trajectory_lnn[i-1,:] + intODE.dt*np.array([trajectory_lnn[i-1,0], trajectory_lnn[i-1,1],lnn_net_acc[0], lnn_net_acc[1]])

            
        # Plot a trajectory of double pendulum:
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(6, 8))
        axs[0].plot(t, trajectory[:,0])
        axs[0].plot(t, trajectory_net[:,0])
        axs[0].plot(t, trajectory_lnn[:,0])
        axs[0].set_ylabel("Pole 1 Angle (rad)")
        axs[1].plot(t,trajectory[:,1])
        axs[1].plot(t,trajectory_net[:,1])
        axs[1].plot(t,trajectory_lnn[:,1])
        axs[1].set_ylabel("Pole 2 Angle (rad)")
        axs[2].plot(t, trajectory[:,2])
        axs[2].plot(t, trajectory_net[:,2])
        axs[2].plot(t, trajectory_lnn[:,2])
        axs[2].set_ylabel("Pole 1 Angular velocity (rad/s)")
        axs[3].plot(t,trajectory[:,3])
        axs[3].plot(t,trajectory_net[:,3])
        axs[3].plot(t,trajectory_lnn[:,3])
        axs[3].set_ylabel("Pole 2 Angular velocity (rad/s)")
        axs[3].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()


        # fig = plt.figure()
        # images =dp.get_dblpend_images(trajectory,fig,ax=fig.gca(), l1=1, l2=1, verbose=False)
        # fig2image = plt.imshow(np.hstack(images))
        # plt.axis('off')
        # plt.show()



 # Works ...
# lnn_lg = LN_net(input)
# grad_output = torch.autograd.grad(lnn_lg, input, create_graph=True)[0]
# hessian = []
# for grad_elem in grad_output.view(-1):
#     hessian_row = torch.autograd.grad(grad_elem, input, retain_graph=True)[0]
#     hessian.append(hessian_row.view(-1))

# hessian = torch.stack(hessian)
# print(hessian)




# tried ....
# gradient = torch.autograd.grad(lnn_lg, input, create_graph=True)[0]
# hessian = torch.zeros(input.size()[0], input.size()[0]) 

# for i in range(input.size()[0]):
#     # Compute the second derivative of each element of the gradient vector
#     grad_i = gradient[i]
#     hessian[i] = torch.autograd.grad(grad_i, input, retain_graph=True)[0][0]

# print("Hessian matrix:")
# print(hessian)

# tried ....
# lnn_lg.backward(torch.ones_like(lnn_lg), retain_graph=True)
# print(f"First call\n{input.grad}")
# print(input._grad_fn)

# tried ....
# jacobian_func = lambda x: torch.autograd.functional.jacobian(LN_net, x, create_graph=True)

# grad = torch.autograd.functional.jacobian(LN_net, torch.tensor(trajectory[i-1,:], requires_grad=True, dtype=torch.float), create_graph=True, vectorize=True).detach().numpy()
# hess = grad.T @ grad
# hess = hessian(LN_net, torch.tensor(trajectory[i-1,:], requires_grad=True, dtype=torch.float))
# print(hess)

# hess = LN_net.hessian(LN_net(torch.tensor(trajectory[i-1,:], requires_grad=True, dtype=torch.float)), 
#                       torch.tensor(trajectory[i-1,:], requires_grad=True, dtype=torch.float).float())



# lnn_acc = 
        

        


