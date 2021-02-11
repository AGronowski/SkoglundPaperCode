import torch 
import torchvision 
import random 
import numpy as np 
import pandas as pd
from sklearn import preprocessing
import argparse

torch.manual_seed(2020)
np.random.seed(2020)

def weight_init(m):
    '''
    This function is used to initialize the netwok weights
    '''

    if isinstance(m,torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m,torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)

class MNIST_dataset(torch.utils.data.Dataset):

    def __init__(self,data,targets,data_hidden,transform=None,task='privacy'):
        #data is (num_examples, 3, 28, 2) with 1 of the 3 rows containing the image data, rest 0s
        #data hidden is list of numbers 0,1,2
        self.data = data
        self.targets = targets
        self.hidden = data_hidden
        self.transform = transform
        self.task = task
        self.target_vals = 10
        self.hidden_vals = 3
    
    def __getitem__(self,index):
        image, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        #pil-python imaging library
        image, target, hidden = torchvision.transforms.functional.to_pil_image(image), int(target), int(hidden)
        #trainsform is trainset.transform / testset.transform
        if self.transform is not None:
            image = self.transform(image)
        if self.task == 'fairness':
            return image, target, hidden
        else: 
            return image, image, hidden
    
    def __len__(self):
        return len(self.targets)

def get_mnist(task='privacy'):

    # Load normal MNIST dataset 
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))
    
    # Add the color and normalize to 0..1
    N_tr = len(trainset)
    data_n = torch.zeros(N_tr,3,28,28)
    data_hidden = torch.arange(len(trainset.targets)) % 3 #list of numbers from 0 to 2
    #mnist data gets added to either row 0, 1, or 2
    for n in range(N_tr):
        data_n[n,data_hidden[n]] = trainset.data[n] #shape (28,28)
    data_n /= 255.0 
    trainset = MNIST_dataset(data_n,trainset.targets,data_hidden,trainset.transform,task)

    N_tst = len(testset)
    data_n = torch.zeros(N_tst,3,28,28)
    data_hidden = torch.arange(len(testset.targets)) % 3
    for n in range(N_tst):
        data_n[n,data_hidden[n]] = testset.data[n]
    data_n /= 255.0 
    testset = MNIST_dataset(data_n,testset.targets,data_hidden,testset.transform,task)

    return trainset, testset

class Adult_dataset(torch.utils.data.Dataset):

    def __init__(self,data,targets,data_hidden,transform=None,task='fairness'):
        self.data = data #X
        self.targets = targets #T
        self.hidden = data_hidden #S
        self.transform = transform #this is none
        self.target_vals = 2
        self.hidden_vals = 2
        self.task = task
    
    def __getitem__(self,index):
        datum, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        if self.task == 'fairness':
            target, hidden = int(target), int(hidden)
        else: 
            hidden = int(hidden)
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, target, hidden
    
    def __len__(self):
        return len(self.targets)

def get_adult(task='fairness'):

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country','salary']

    #these are the categorical variables that will be encoded
    dummy_variables = {
        'workclass': ['Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'],
        'education': ['Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, \
            12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'],
        'education-num' : ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'],
        'marital-status': ['Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent,\
            Married-AF-spouse'],
        'occupation': ['Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, \
            Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, \
            Protective-serv, Armed-Forces'],
        'relationship': ['Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'],
        'race': ['White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'],
        'sex': ['Female, Male'],
        'native-country' : ['United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), \
            India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, \
            Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, \
            Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands']
    }

    #break apart each string into a list based on the comma separator then strip whitespace
    for k in dummy_variables:
        dummy_variables[k] = [v.strip() for v in dummy_variables[k][0].split(',')]
    
    # Load Adult dataset

    # This should be uncommented first time this is run to download the datasets
    # This code doesn't save the dataset, I downloaded it manually to data folder
    '''
    data_train = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        names=column_names,header=None
    )
    data_test = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        names=column_names,skiprows=1,header=None
    ) 
    '''

    data_train = pd.read_csv(
        '../data/adult_us_cens/adult.data',
        names=column_names,header=None #names creates header row
    )
    data_test = pd.read_csv(
        '../data/adult_us_cens/adult.test',
        names=column_names,skiprows=1,header=None
    )#skip 1st row

    #apply function to each column
    #removes whitespace
    data_train = data_train.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
    data_test = data_test.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)

    def get_variables(data, task='fairness'):

        #Encode target labels with value between 0 and n_classes-1
        le = preprocessing.LabelEncoder()
        dummy_columns = list(dummy_variables.keys())
        dummy_columns.remove('sex')
        #integer encodings to the dummy columns
        data[dummy_columns] = data[dummy_columns].apply(lambda col: le.fit_transform(col))
        X = data.drop('sex',axis=1).drop('salary',axis=1).to_numpy().astype(float)
        S = data['sex'].to_numpy()
        if task=='fairness':
            T = data['salary'].to_numpy()
            #return 0 when T <=50K, 1 otherwis
            T = np.where(np.logical_or(T=='<=50K',T=='<=50K.'),0,1)
        else:
            T = data.drop('sex',axis=1).drop('salary',axis=1).to_numpy().astype(float)
        S = np.where(S=='Male',0,1)

        return X, S, T
    
    X_train, S_train, T_train = get_variables(data_train,task)
    X_test, S_test, T_test = get_variables(data_test,task)
    #mean and std for each column (reduce rowns)
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    #normalize data
    X_train = (X_train-X_mean) / (X_std)
    X_test = (X_test-X_mean) / (X_std)
    if task == 'privacy':
        for i in range(len(T_train[1,:])):
            if len(np.unique(T_train[:,i])) > 42:
                t_mean, t_std = T_train[:,i].mean(), T_train[:,i].std() 
                T_train[:,i] = (T_train[:,i]-t_mean) / t_std
                T_test[:,i] = (T_test[:,i]-t_mean) / t_std

    trainset = Adult_dataset(X_train, T_train, S_train, task=task)
    testset = Adult_dataset(X_test, T_test, S_test, task=task)

    return trainset, testset


class Mh_dataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, data_hidden, transform=None, task='fairness'):
        self.data = data #X
        self.targets = targets #T
        self.hidden = data_hidden #S
        self.transform = transform
        self.target_vals = 2
        self.hidden_vals = 2
        self.task = task

    def __getitem__(self, index):
        datum, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        if self.task == 'fairness':
            target, hidden = int(target), int(hidden)
        else:
            hidden = int(hidden)
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, target, hidden

    def __len__(self):
        return len(self.targets)

def get_mh(task='fairness'):
    # header is removed automatically (header=0)
    data = pd.read_csv('../data/mental_health/imputed_cleaned_mental_health.csv')

    # drop all rows where Gender is other. inplace=True means it's permanently removed from the original df
    # we now have 1402 examples instead of 1428
    data.drop(data[data['Gender'] == "other"].index, inplace=True)

    columns = ['mh_coverage_flag', 'mh_resources_provided', 'mh_anonimity_flag',
                    'mh_prod_impact', 'mh_medical_leave', 'mh_discussion_neg_impact',
                    'mh_family_hist', 'mh_disorder_past', 'AgeBinary', 'Gender','treatment']

    # drop all columns except for feature_cols
    data = data[columns]

    # replace the binary features with 0s and 1s
    data['Gender'] = data['Gender'].replace(to_replace=['male','female'],value=[1,0])
    data['AgeBinary'] = data['AgeBinary'].replace(to_replace=['< 40yo','>= 40yo'],value=[1,0])

    # these are categorical features that will be encoded
    dummy_variables = ['mh_coverage_flag', 'mh_resources_provided', 'mh_anonimity_flag',
                    'mh_prod_impact', 'mh_medical_leave', 'mh_discussion_neg_impact',
                    'mh_family_hist', 'mh_disorder_past']

    #train test split
    msk = np.zeros(len(data))
    #1st 70% of the list are 1s
    msk[:int(0.7*len(data))] = 1
    #permute the 1s and 0s at random
    msk = np.random.permutation(msk).astype('bool')
    data_train = data[msk]
    data_test = data[~msk]


    def get_variables(data, task='fairness'):

        #Encode target labels with value between 0 and n_classes-1
        le = preprocessing.LabelEncoder()

        #integer encodings to the dummy columns
        data = data.copy() #taking a coopy prevents warnings
        data[dummy_variables] = data[dummy_variables].apply(lambda col: le.fit_transform(col))
        X = data.drop('Gender',axis=1).drop('treatment',axis=1).to_numpy().astype(float)
        S = data['Gender'].to_numpy()
        if task=='fairness':
            T = data['treatment'].to_numpy()
        # else:
        #     T = data.drop('sex',axis=1).drop('salary',axis=1).to_numpy().astype(float)

        return X, S, T

    X_train, S_train, T_train = get_variables(data_train,task)
    X_test, S_test, T_test = get_variables(data_test,task)
    #mean and std for each column (0 means column) (reduce rowns)
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    #normalize data
    X_train = (X_train-X_mean) / (X_std)
    X_test = (X_test-X_mean) / (X_std)

    trainset = Adult_dataset(X_train, T_train, S_train, task=task)
    testset = Adult_dataset(X_test, T_test, S_test, task=task)

    return trainset, testset

class Compas_dataset(torch.utils.data.Dataset):

    def __init__(self,data,targets,data_hidden,transform=None,task='fairness'):
        self.data = data
        self.targets = targets
        self.hidden = data_hidden
        self.transform = transform
        self.target_vals = 2
        self.hidden_vals = 2
        self.task = task
    
    def __getitem__(self,index):
        datum, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        if self.task == 'fairness':
            target, hidden = int(target), int(hidden)
        else: 
            hidden = int(hidden)
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, target, hidden
    
    def __len__(self):
        return len(self.targets)

def get_compas(task='fairness'):

    data = pd.read_csv(
        '../data/compas/propublica_data_for_fairml.csv',
        header=0,sep=','
    )

    msk = np.zeros(len(data))
    #train test split
    #1st 70% of the list are 1s
    msk[:int(0.7*len(data))] = 1
    msk = np.random.permutation(msk).astype('bool')
    data_train = data[msk] 
    data_test = data[~msk]

    def get_variables(_data):
        
        X = _data.drop('Two_yr_Recidivism',axis=1).drop('African_American',axis=1).to_numpy()
        S = _data['African_American'].to_numpy()
        T = _data['Two_yr_Recidivism'].to_numpy()
        
        return X, S, T

    X_train, S_train, T_train = get_variables(data_train)
    X_test, S_test, T_test = get_variables(data_test)

    if task == 'privacy':
        T_train = X_train
        T_test = X_test
    
    mean, std = X_train[:,0].mean(), X_train[:,0].std()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    trainset = Compas_dataset(X_train, T_train, S_train, task=task)
    testset = Compas_dataset(X_test, T_test, S_test, task=task)

    return trainset, testset

# return experiment value from command line
def get_args():
    parser = argparse.ArgumentParser(
        description = 'Run the variational approach to the CPF or the CFB',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--experiment', type = int, default = 1, 
        help = 'Experiment to perform. Meaning:\n \
                1 - Example on the Colored MNIST dataset\n \
                2 - Fairness on the Adult dataset\n \
                3 - Fairness on the Colored MNIST dataset\n \
                4 - Privacy on the Adult dataset\n \
                5 - Privacy on the Colored MNIST dataset\n \
                6 - Fairness on the COMPAS dataset\n \
                7 - Privacy on the COMPAS dataset\n')

    return parser.parse_args()