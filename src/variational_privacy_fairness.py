import torch, torchvision 
from progressbar import progressbar
from networks import Encoder, Decoder, Decoder2
import metrics
from mutual_information import get_conditional_entropy, get_entropy
from kde_estimation import KDE_IXY_estimation
import math, os 
import umap 
import warnings 
from numba.errors import NumbaPerformanceWarning 
from visualization import plot_embeddings

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

torch.manual_seed(2020)

class VPAF(torch.nn.Module):

    def __init__(self, input_type='image', representation_type='image', output_type=['image'], s_type='classes', input_dim=104, \
            representation_dim=8, output_dim=[1], s_dim=1, problem='privacy', beta=0.0, beta2=0.0, gamma=1.0, prior_type='Gaussian'):
        super(VPAF,self).__init__() 

        self.problem = problem   
        self.param = gamma if self.problem == 'privacy' else beta
        self.param2 = beta2
        self.input_type = input_type  #vector or image
        self.representation_type = representation_type #vector for mh
        self.output_type = output_type  #['binary'] for mh
        self.output_dim = output_dim #[1] for mh
        self.s_type = s_type #'classes'
        self.prior_type=prior_type #'Gaussian'

        self.encoder = Encoder(input_type, representation_type, input_dim, representation_dim) 
        self.decoder = Decoder(representation_type, output_type, representation_dim, output_dim, s_dim=s_dim)
        self.decoder2 = Decoder2(representation_dim,output_dim) #rep_dim = 2, output_dim = [1] for mh
    
    def get_IXY_ub(self, y_mean, mode='Gaussian'):

        if mode == 'Gaussian':

            #ymean is encoder output
            #self.encoder.y_logvar_theta is ??? starts at -1, then is optimized by gradient descent

            #Z is two dimensional
            #y_mean, y_mean^w is 981 x 2 tenspr
            #others are shape 1
            #sum sums each component in the 981 x 2 tensor giving 1 scalar
            #
            Dkl = -0.5 * torch.sum(
                1.0 + self.encoder.y_logvar_theta - torch.pow(y_mean, 2) - torch.exp(self.encoder.y_logvar_theta)
            )
            IXY_ub = Dkl / math.log(2) # in bits 
        else: # MoG
            IXY_ub = KDE_IXY_estimation(self.encoder.y_logvar_theta, y_mean)
            IXY_ub /= math.log(2) # in bits

        return IXY_ub

    def get_H_output_given_SY_ub(self, decoder_output, t):

        #for adult is true, t is binary tensor length 981
        if len(t.shape) == 1:
            t = t.view(-1,1) #t now becomes 2d tensor

        H_output_given_SY_ub = 0
        dim_start_out = 0
        dim_start_t = 0
        reg_start = 0

        #zip creates tuples of the two inputs
        #(binary, 1)
        for output_type_, output_dim_ in zip(self.output_type,self.output_dim):

            if output_type_ == 'classes':
                so = dim_start_out
                eo = dim_start_out + output_dim_
                st = dim_start_t
                et = dim_start_t + 1
                #tensor.long() converts to 32 bit int

                CE = torch.nn.functional.cross_entropy(decoder_output[:,so:eo], t[:,st:et].long().view(-1), reduction='sum')
            elif output_type_ == 'binary': #true for adult
                so = dim_start_out
                eo = dim_start_out + 1
                st = dim_start_t
                et = dim_start_t + 1
                #tensor.view(-1) reshapes tensor with unknown number of rows
                # t is the target Y
                #t [:,0:1] selects all in 981 x 1 tensor
                #.view(-1) changes back to 1D tensor
                CE = torch.nn.functional.binary_cross_entropy_with_logits(decoder_output[:,so:eo].view(-1), t[:,st:et].view(-1), reduction='sum')
            elif output_type_ == 'image':
                eo = et = 0
                CE = torch.nn.functional.binary_cross_entropy(decoder_output, t, reduction='sum')
            else: # regression 
                so = dim_start_out
                eo = dim_start_out + output_dim_
                st = dim_start_t
                et = dim_start_t + output_dim_
                sr = reg_start 
                er = reg_start + output_dim_
                reg_start = er
                CE = 0.5 * torch.sum(
                    math.log(2*math.pi) + self.decoder.out_logvar_phi[sr:er] + \
                        torch.pow(decoder_output[:,so:eo] - t[:,st:et], 2) / (torch.exp(self.decoder.out_logvar_phi[sr:er]) + 1e-10)
                )

            H_output_given_SY_ub += CE / math.log(2) # in bits 

            dim_start_out = eo
            dim_start_t = et

        return H_output_given_SY_ub


    def evaluate_privacy(self, dataloader, device, N, batch_size, figs_dir, verbose):

        IXY_ub = 0
        H_X_given_SY_ub = 0

        with torch.no_grad():
            for it, (x, t, s) in enumerate(dataloader):

                x = x.to(device).float() 
                t = t.to(device).float() 
                s = s.to(device).float()

                #intermediate representation Z
                y, y_mean = self.encoder(x)
                #prediction Y based on Z and A
                output = self.decoder(y,s) 

                if self.input_type == 'image' and self.representation_type == 'image' and 'image' in self.output_type and it == 1:
                    torchvision.utils.save_image(x[:12*8],os.path.join(figs_dir,'x.eps'),nrow=12)
                    torchvision.utils.save_image(y_mean[:12*8],os.path.join(figs_dir,'y.eps'),nrow=12)
                    torchvision.utils.save_image(output[:12*8],os.path.join(figs_dir,'x_hat.eps'),nrow=12)
                
                IXY_ub += self.get_IXY_ub(y_mean, self.prior_type)
                H_X_given_SY_ub += self.get_H_output_given_SY_ub(output, t)
                if self.representation_type == 'image':
                    if it == 0 and self.s_type == 'classes':
                        reducer_y = umap.UMAP(random_state=0)
                        reducer_y.fit(y_mean.cpu().view(batch_size,-1),y=s.cpu())
                        reducer_x = umap.UMAP(random_state=0)
                        reducer_x.fit(x.cpu().view(batch_size,-1),y=s.cpu())
                    if it == 1:
                        if self.s_type == 'classes':
                            embedding_s_y = reducer_y.transform(y_mean.cpu().view(batch_size,-1))
                            embedding_s_x = reducer_x.transform(x.cpu().view(batch_size,-1))
                        reducer_y = umap.UMAP(random_state=0)
                        reducer_x = umap.UMAP(random_state=0)
                        embedding_y = reducer_y.fit_transform(y_mean.cpu().view(batch_size,-1))
                        embedding_x = reducer_x.fit_transform(x.cpu().view(batch_size,-1))
                        if self.s_type == 'classes':
                            plot_embeddings(embedding_y, embedding_s_y, s.cpu().view(batch_size).long(), figs_dir, 'y')
                            plot_embeddings(embedding_x, embedding_s_x, s.cpu().view(batch_size).long(), figs_dir, 'x')
                        else: 
                            plot_embeddings(embedding_y, embedding_y, -1, figs_dir, 'y')
                            plot_embeddings(embedding_y, embedding_y, -1, figs_dir, 'x')
        
        IXY_ub /= N
        H_X_given_SY_ub /= N          
        print(f'IXY: {IXY_ub.item()}') if verbose else 0 
        print(f'HX_given_SY: {H_X_given_SY_ub.item()}') if verbose else 0
        return IXY_ub, H_X_given_SY_ub
    
    def evaluate_fairness(self, dataloader, device, N, target_vals, H_T_given_S, verbose):

        IXY_ub = 0
        H_T_given_SY_ub = 0
        accuracy = 0

        with torch.no_grad():
            for it, (x, t, s) in enumerate(dataloader):

                x = x.to(device).float() 
                t = t.to(device).float() 
                s = s.to(device).float() 

                y, y_mean = self.encoder(x) 
                output = self.decoder(y,s) 
                
                IXY_ub += self.get_IXY_ub(y_mean, self.prior_type)
                H_T_given_SY_ub += self.get_H_output_given_SY_ub(output, t)
                accuracy += metrics.get_accuracy(output, t, target_vals) * len(x) 
                
        
        IXY_ub /= N
        H_T_given_SY_ub /= N 
        print(H_T_given_SY_ub)
        accuracy /= N   
        IYT_given_S_lb = H_T_given_S - H_T_given_SY_ub.item()
        print(f'I(X;Y) = {IXY_ub.item()}') if verbose else 0 
        print(f'I(Y;T|S) = {IYT_given_S_lb}') if verbose else 0
        print(f'Accuracy (network): {accuracy}') if verbose else 0 
        return IXY_ub, IYT_given_S_lb

    def evaluate(self, dataset, verbose, figs_dir):

        device = 'cuda' if next(self.encoder.parameters()).is_cuda else 'cpu' 
        batch_size = 2048 
        if len(dataset) < 2048:
            batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
        if self.problem == 'privacy':
            IXY_ub, H_X_given_SY_ub = self.evaluate_privacy(dataloader, device, len(dataset), batch_size, figs_dir, verbose)
            return IXY_ub, H_X_given_SY_ub
        else: # fairness
            H_T_given_S = get_conditional_entropy(dataset.targets, dataset.hidden, dataset.target_vals, dataset.hidden_vals)
            IXY_ub, IYT_given_S_lb = self.evaluate_fairness(dataloader, device, len(dataset), dataset.target_vals, H_T_given_S, verbose)
            return IXY_ub, IYT_given_S_lb

    def train_step(self, dataloader, optimizer, verbose):

        device = 'cuda' if next(self.encoder.parameters()).is_cuda else 'cpu' 

        #get minibatches of x ,t, s
        for x, t, s in progressbar(dataloader):

            x = x.to(device).float() 
            t = t.to(device).float()
            s = s.to(device).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # intermediate representation Z
            # y has noise from standard normal distribution added to it
            y, y_mean = self.encoder(x) #calls forward method in encoder
            #compression term to be minimized
            IXY_ub = self.get_IXY_ub(y_mean, self.prior_type)

            #negative of accuracy term to be maximized

            # prediction Y based on Z and A
            output = self.decoder(y,s) #calls forward method in decoder
            #this is simply binary cross-entropy (negative of the lower bound)
            H_output_given_SY_ub = self.get_H_output_given_SY_ub(output, t) #t is target - true label

            #prediction Y based just on Z
            output2 = self.decoder2(y)
            H_output_given_SY_ub_2 = self.get_H_output_given_SY_ub(output2, t)

            #loss functionb
            #param is Beta - values from 1 to 50

            #loss = IXY_ub - self.param * H_output_given_SY_ub - self.param2 * H_output_given_SY_ub_2
            loss = IXY_ub + self.param * H_output_given_SY_ub
            #backward propagation
            loss.backward()
            #initiate gradient descent
            optimizer.step()


    #train the network
    def fit(self, dataset_train, dataset_val, epochs=1000, learning_rate=1e-3, batch_size=1024, eval_rate=15, \
        verbose=True, logs_dir='../results/logs/', figs_dir='../results/images/'):

        dataloader = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        for epoch in range(epochs): 
            print(f'Epoch # {epoch+1}')

            #backward propagation and gradient descent
            self.train_step(dataloader, optimizer, verbose)

            if epoch % eval_rate == eval_rate - 1:
                print(f'Evaluating TRAIN') if verbose else 0 
                if self.problem == 'privacy':
                    IXY_ub, H_X_given_SY_ub = self.evaluate(dataset_train, verbose, figs_dir)
                else: # fairness
                    self.evaluate(dataset_train, verbose, figs_dir)
                    print(f'Evaluating VALIDATION/TEST') if verbose else 0 
                    XY_ub, IYT_given_S_lb = self.evaluate(dataset_val, verbose, figs_dir)
