import torch, torchvision
import math
import utils
import progressbar
import numpy as np

torch.manual_seed(0)

class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size = 100):
        super(MLP, self).__init__()
    
        self.f_theta = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_size, 1),
        )
    
    def forward(self, X, Y):

        Z = torch.cat((X,Y),1)
        return self.f_theta(Z)

class CNN(torch.nn.Module):

    def __init__(self, dimX, dimY):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,5,5,1,padding=1),
            torch.nn.MaxPool2d(5,2,2),
            torch.nn.ReLU6(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(5,50,5),
            torch.nn.MaxPool2d(5,2,2),
            torch.nn.ReLU6(inplace=True),
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(50*5*5 + 1,100),
            torch.nn.ReLU6(inplace=True),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(100+1,50),
            torch.nn.ReLU6(inplace=True)
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(50+1,1)
        )
    
    def forward(self, X, Y):

        Y = Y.view(-1,1)
        Z = self.conv1(X + Y.view((-1,) + (1,)*(len(X.shape)-1)))
        Z = self.conv2(Z + Y.view((-1,) + (1,)*(len(Z.shape)-1)))
        Z = self.fc1(torch.cat([Z.flatten(1),Y],1))
        Z = self.fc2(torch.cat([Z,Y],1))
        return self.fc3(torch.cat([Z,Y],1))

class MINE(torch.nn.Module):

    def __init__(self, dimX, dimY, moving_average_rate = 0.01, hidden_size = 100, network_type = 'mlp'):
        super(MINE, self).__init__()

        self.dimX = dimX 
        self.dimY = dimY
        if network_type == 'mlp':
            self.network = MLP(self.dimX + self.dimY, hidden_size)
        elif network_type == 'cnn':
            self.network = CNN(self.dimX, self.dimY)
        self.network.apply(utils.weight_init)
        self.moving_average_rate = moving_average_rate
    
    def get_mi(self, X, Y, Y_tilde):

        T = self.network(X, Y).mean()
        expT = torch.exp(self.network(X, Y_tilde)).mean()
        mi = (T - torch.log(expT)).item() / math.log(2)
        return mi, T, expT
    
    def sample_batch(self, data_X, data_Y, batch_size):

        index_joint = np.random.choice(len(data_X),size=batch_size,replace=False)
        index_marginal = np.random.choice(len(data_X),size=batch_size,replace=False)
        return data_X[index_joint], data_Y[index_joint], data_Y[index_marginal]

    
    def train(self, data_X, data_Y, learning_rate = 1e-3, batch_size = 256, n_iterations = int(5e3), n_verbose = 1000, n_window = 100, decay_rate = 0.9, n_decay = -1, save_progress=200):

        if torch.is_tensor(data_X):
            data_X = data_X.view(-1,self.dimX).float() if self.dimX == 1 else data_X.float()
        else: 
            data_X = torch.FloatTensor(data_X).view(-1,self.dimX) if self.dimX == 1 else torch.FloatTensor(data_X)
        if torch.is_tensor(data_Y):
            data_Y = data_Y.view(-1,self.dimY).float() if self.dimY == 1 else data_Y.float()
        else:
            data_Y = torch.FloatTensor(data_Y).view(-1,self.dimY) if self.dimY == 1 else torch.FloatTensor(data_Y)

        device = 'cuda' if next(self.network.parameters()).is_cuda else 'cpu'

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        if n_decay > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,n_decay,decay_rate)

        iteration = 0
        moving_average_expT = 1
        mi = torch.empty(n_window)

        if save_progress > 0:
           mi_progress = torch.zeros(int(n_iterations / save_progress)) 

        for iteration in progressbar.progressbar(range(n_iterations)):

            X, Y, Y_tilde = self.sample_batch(data_X,data_Y,batch_size)
            X = torch.autograd.Variable(X).to(device)
            Y = torch.autograd.Variable(Y).to(device)
            Y_tilde = torch.autograd.Variable(Y_tilde).to(device)

            mi_lb, T, expT = self.get_mi(X, Y, Y_tilde)
            moving_average_expT = ((1-self.moving_average_rate) * moving_average_expT + self.moving_average_rate * expT).item()
            loss = -1.0 * (T - expT / moving_average_expT)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mi[iteration % n_window] = mi_lb
                if iteration >= n_window and iteration % n_verbose == n_verbose - 1:
                    print(f'Iteration {iteration+1}: {mi.mean().item()}')
                
                if save_progress > 0 and iteration % save_progress == save_progress - 1:
                    mi_progress[int(iteration / save_progress)] = mi.mean().item()
            
                if iteration % n_decay == n_decay - 1:
                    scheduler.step()

        if save_progress > 0:
            return mi_progress

        return mi.mean().item()
