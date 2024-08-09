import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime as dt
import multiprocessing
import abc

from ncps.torch import LTC
from ncps.wirings import FullyConnected
import traceback 
from sklearn.model_selection import KFold

from LTC_learner import SequenceLearner,ScheduledSamplingSequenceLearner

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, BatchSampler, SequentialSampler, Sampler, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from optuna.integration import PyTorchLightningPruningCallback

    
class BatchShuffleSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        # Get all indices from the sampler
        indices = list(self.sampler)
        # Calculate the number of batches
        num_batches = len(indices) // self.batch_size
        if not self.drop_last:
            num_batches += 1 if len(indices) % self.batch_size != 0 else 0

        # Create batch indices
        # batch_indices = [indices[batch_start,batch_start+self.batch_size] for batch_start in range(0,indices,self.batch_size)]
        batch_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(num_batches)]
        # Shuffle the order of batches, not the data within batches
        np.random.shuffle(batch_indices)

        # Flatten the list of batch indices
        shuffled_indices = [batch for batch in batch_indices]
        return iter(shuffled_indices)

    def __len__(self):
        return len(self.sampler) // self.batch_size if self.drop_last else (len(self.sampler) + self.batch_size - 1) // self.batch_size

class BatchWeightedShuffleSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last,item_weights,gen,replacement=True):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.item_weights = item_weights
        self.replacement = replacement
        self.gen = gen
        

    def __iter__(self):
        # Get all indices from the sampler
        indices = list(self.sampler)
        # Calculate the number of batches
        num_batches = len(indices) // self.batch_size
        if not self.drop_last:
            num_batches += 1 if len(indices) % self.batch_size != 0 else 0
        batch_weights = torch.tensor([sum(self.item_weights[i*self.batch_size:(i+1)*self.batch_size]) for i in range(num_batches)])
        batch_weights = batch_weights / torch.sum(batch_weights)
        # select how we sample batches
        selected_batches = iter(torch.multinomial(torch.tensor(batch_weights,dtype=float), num_batches, self.replacement,generator=self.gen).tolist())
        # Create batch indices
        batch_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in selected_batches]
        # Shuffle the order of batches, not the data within batches
        np.random.shuffle(batch_indices)

        # Flatten the list of batch indices
        shuffled_indices = [batch for batch in batch_indices]
        return iter(shuffled_indices)

    def __len__(self):
        return len(self.sampler) // self.batch_size if self.drop_last else (len(self.sampler) + self.batch_size - 1) // self.batch_size
    
class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, inputs, targets):
        # normalzed mean squared error
        loss = torch.mean(((targets-inputs)**2))/ torch.mean(targets)
        return loss
    
class MSELossfuture(nn.Module):
    def __init__(self, n_future):
        super(MSELossfuture, self).__init__()
        self.n_future = n_future
        self.loss = torch.nn.MSELoss()
        print(" using loss on future only")
    def forward(self, inputs, targets):
        # normalzed mean squared error
        loss = self.loss(inputs[:,-self.n_future:] ,targets[:,-self.n_future:])
        return loss

def load_trace():
    df = pd.read_csv("data/traffic/Metro_Interstate_Traffic_Volume.csv")
    holiday = (df["holiday"].values == None).astype(np.float32)
    temp = df["temp"].values.astype(np.float32)
    # temp -= np.mean(temp)  # normalize temp by annual mean
    rain = df["rain_1h"].values.astype(np.float32)
    snow = df["snow_1h"].values.astype(np.float32)
    clouds = df["clouds_all"].values.astype(np.float32)
    date_time = df["date_time"].values
    # 2012-10-02 13:00:00
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time]
    weekday = np.array([d.weekday() for d in date_time]).astype(np.float32)
    noon = np.array([d.hour for d in date_time]).astype(np.float32)
    noon = np.sin(noon * np.pi / 24)

    features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon], axis=-1)

    traffic_volume = df["traffic_volume"].values.astype(np.float32)
    # traffic_volume -= np.mean(traffic_volume)  # normalize
    # traffic_volume /= np.std(traffic_volume)  # normalize

    return features, traffic_volume

def cut_in_sequences(x,seq_len,inc=1,prognosis=1):
    sequences_x = []
    sequences_y = []
    # x: time series
    # y: time series shifted into futurew
    # every x array has an overlap of (seq_len - inc)
    for s in range(0,x.shape[0] - seq_len-prognosis,inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(x[start+prognosis:end+prognosis])
        # sequences_y.append(x[end:end+prognosis])
    return sequences_x,sequences_y

class DataBaseClass:
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self.use_validation_data = True
    
    @abc.abstractmethod 
    def normalize_data(self):
        dataset_names = ["train_x","train_y","valid_x", "valid_y","test_x","test_y","test_plot_x","test_plot_y"]
        train_with_noise = False

        for _name in dataset_names:
            _array = getattr(self,_name)
            _array = self.normalize(_array,_name.split("_")[-1])
            print(f"{_name} shape: ", str(_array.shape),f"{_name} post normalisation: ",_array.mean())
            if "train" in _name and train_with_noise:
                noise = np.random.normal(0, 0.1, _array.shape)   
                _array = _array + noise
                print(f"{_name} post noise: ",str(_array.mean()))              
            setattr(self,_name, _array)
            

    @abc.abstractmethod 
    def get_dataloader(self,subset="train"):
        """ We create dataloader with content of shapes [batch size,series_length,features]
        permute x and y data because [series_length, n_series, features]
        """

        # the predict dataset is for now made from the test chunk and we use it to make plots
        #　for now the predict dataset only exists for neuron data 
        if subset == "predict":
            """expect x and y to be in format (seq_len,1,neurons) for normalisation"""
            # self.load_realtime_data()
            return self.normalize(np.expand_dims(self.predict_x,0),"x"), self.predict_x 
        
        assert (subset) in ["train","valid","test","test_plot"]
        x_data = torch.permute(getattr(self,f"{subset}_x"),(1,0,2)) # _batch = (B_size, Timesteps, Features)
        y_data = torch.permute(getattr(self,f"{subset}_y"),(1,0,2))
        dataset = TensorDataset(x_data, y_data)

        num_workers = 4
        batch_size = self.batch_size
        weights =  (x_data[:, -5:,-1] == torch.max(x_data[...,-1]))
        weights = torch.sum(weights,dim=1)
        weights = (weights* (self.batch_size*4)) + 1
        generator = torch.manual_seed(928405)

        if self.scheduled_sampling and "train" in subset:
            sequential_sampler = SequentialSampler(dataset)   
            # batch_sampler = BatchShuffleSampler(sequential_sampler,batch_size=self.batch_size, drop_last=False)
            batch_sampler = BatchWeightedShuffleSampler(sequential_sampler,batch_size=self.batch_size,drop_last=False,item_weights=weights,gen = generator)
            return DataLoader(
                dataset = dataset,
                batch_sampler= batch_sampler,
                num_workers = num_workers
            )
        elif "train" in subset:
            weights = weights / torch.sum(weights)
            sampler = WeightedRandomSampler(weights,x_data.shape[0],replacement=True,generator=generator)
            return DataLoader(
                    dataset = dataset,
                    batch_size =  batch_size,
                    num_workers = num_workers,
                    sampler = sampler
                )
        elif not "train" in subset:
            if subset == "test_plot" and self.iterative_forecast:
                batch_size = self.n_forecasts 
                num_workers = 1
            return DataLoader(
                shuffle = False,
                dataset = dataset,
                batch_size =  batch_size,
                num_workers = num_workers,
            )


    @abc.abstractmethod 
    def normalize(self, _tensor,values="x"):
        _tensor = torch.tensor(_tensor)
        if not hasattr(self,"std") or not hasattr(self,"mean"):
            self.mean = {}
            self.std = {}
        if not values in self.mean or not values in self.std:
            self.mean[values] = torch.mean(_tensor, dim=(0, 1), keepdim=True)
            self.std[values] = torch.std(_tensor, dim=(0, 1), keepdim=True)
            res = self.std[values].clone()
            res[self.std[values]==0] = torch.tensor(1)
            self.std[values] = res
            del res
        #because if we use realtime data we do not have the laser-activation data channel in the input x data so this is not normalized
        return (_tensor - self.mean[values][...,:_tensor.shape[-1]]) / self.std[values][...,:_tensor.shape[-1]]


    @abc.abstractmethod 
    def denormalize(self,tensor,values="x"):
        return tensor * self.std[values][...,:tensor.shape[-1]] + self.mean[values][...,:tensor.shape[-1]]


    @abc.abstractmethod 
    def make_sequences(self):
        # cut the data into x and y sequences
        dataset_splits = ["train","valid","test"]
        for _split in dataset_splits:
            _array = getattr(self,f"{_split}_chunk")
            if _split == "train":
                temp_x = []
                temp_y = []
                for _arr in _array:
                    try:
                        temp_x_arr,temp_y_arr = cut_in_sequences(_arr,seq_len=self.seq_len, inc=self.increment ,prognosis=self.future)
                    except Exception as e:
                        print(e)
                    temp_x.extend(temp_x_arr) # (N sequences, 32, N neurons)
                    temp_y.extend(temp_y_arr)
            else:
                temp_x,temp_y = cut_in_sequences(_array,seq_len=self.seq_len, inc=self.increment if not _split == "test" else 1,prognosis=self.future)
            temp_x = np.stack(temp_x, axis=1) 
            temp_y = np.stack(temp_y, axis=1)[:,:,:self.out_features] #deselect neuronlaser 
            setattr(self,f"{_split}_x",temp_x)    
            setattr(self,f"{_split}_y",temp_y)
            if _split == "test" :
                # in case of iterative forecast:
                # every batch contains 5 sequences of (x,y) we need the y's for reference and for x we will append the first batch item with model predictions 
                # prognosis (=future) is always 1
                # otherwise: 
                #   prognosis = future. the data to plot will be composed of the predicted items. (hence the increment)
                temp_x,temp_y = cut_in_sequences(_array,seq_len=self.seq_len, inc=self.future,prognosis=self.future)
                temp_x = np.stack(temp_x, axis=1) 
                temp_y = np.stack(temp_y, axis=1)[:,:,:self.out_features]
                setattr(self,f"test_plot_x",temp_x)    
                setattr(self,f"test_plot_y",temp_y)  

    @abc.abstractmethod 
    def set_kfold_splits(self):
        print(f" {self.n_bins // 32} folds of length {self.seq_len} out of {self.n_bins} bins ")
        kf = KFold(n_splits=5,shuffle=False)
        self.kfold_splits = list(kf.split(list(range(self.n_bins//32))))

class TrafficData(DataBaseClass):
    def __init__(self,seq_len=32,future=1,batch_size=16):
        x, y = load_trace()

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.future = future
        """ 
        x: (T, 7)
        train_x after cut : (12041) * (32,7)
        after stack: (T, 32, 7). should be (32,T,7)
        """
        train_x, train_y = cut_in_sequences(x,seq_len=seq_len, inc=30,prognosis=future)
        self.train_x = np.stack(train_x, axis=1) 
        self.train_y = np.stack(train_y, axis=1)

        # make train-test-val split
        total_seqs = self.train_x.shape[1]

        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.valid_x = self.train_x[:,permutation[:valid_size]]
        self.valid_y = self.train_y[:,permutation[:valid_size]]
        self.test_x = self.train_x[:,permutation[valid_size : valid_size + test_size]]
        self.test_y = self.train_y[:,permutation[valid_size : valid_size + test_size]]
        self.train_x = self.train_x[:,permutation[valid_size + test_size :]]
        self.train_y = self.train_y[:,permutation[valid_size + test_size :]]

        self.feature_labels = ['Holiday','Temperature','Rain','Snow','Clouds','Weekday','Noon']
        super().__init__()

class OccupancyData(DataBaseClass):
    def __init__(self,seq_len=32,future=1,batch_size=16):

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.future = future
    
        train_x,train_y = self.read_file("data/occupancy/datatraining.txt")
        test0_x,test0_y = self.read_file("data/occupancy/datatest.txt")
        test1_x,test1_y = self.read_file("data/occupancy/datatest2.txt")

        mean_x = np.mean(train_x,axis=0)
        std_x = np.std(train_x,axis=0)
        train_x = (train_x-mean_x)/std_x
        test0_x = (test0_x-mean_x)/std_x
        test1_x = (test1_x-mean_x)/std_x
        print(len(train_x),train_x[0].shape)
        """
        x: (T, 5)
        train_x after cut : (1009) * (32,5)
        after stack: (32,T,5)
        """
        train_x,train_y = cut_in_sequences(train_x,seq_len=seq_len, inc=8,prognosis=future)
        test0_x,test0_y = cut_in_sequences(test0_x,seq_len=seq_len, inc=8,prognosis=future)
        test1_x,test1_y = cut_in_sequences(test1_x,seq_len=seq_len, inc=8,prognosis=future)
        print(len(train_x),train_x[0].shape)
        train_x = np.stack(train_x,axis=1)
        train_y = np.stack(train_y,axis=1)
        test0_x = np.stack(test0_x,axis=1)
        test0_y = np.stack(test0_y,axis=1)
        test1_x = np.stack(test1_x,axis=1)
        test1_y = np.stack(test1_y,axis=1)

        total_seqs = train_x.shape[1]
        permutation = np.random.RandomState(893429).permutation(total_seqs)
        valid_size = int(0.1*total_seqs)

        self.valid_x = train_x[:,permutation[:valid_size]]
        self.valid_y = train_y[:,permutation[:valid_size]]
        self.train_x = train_x[:,permutation[valid_size:]]
        self.train_y = train_y[:,permutation[valid_size:]]

        self.test_x = np.concatenate([test0_x,test1_x],axis=1)
        self.test_y = np.concatenate([test0_y,test1_y],axis=1)
        self.feature_labels = ['Temperature', 'Humidity','Light','CO2','HumidityRatio']
        super().__init__()

    def read_file(self,filename):
        df = pd.read_csv(filename)                                    

        data_x = np.stack([
            df['Temperature'].values,
            df['Humidity'].values,
            df['Light'].values,
            df['CO2'].values,
            df['HumidityRatio'].values,
            ],axis=-1)
        data_y = df['Occupancy'].values.astype(np.int32)
        return data_x,data_y
            
class CheetahData(DataBaseClass):
    def __init__(self,seq_len=32,future=1,batch_size=16):
        all_files = sorted([os.path.join("data/cheetah",d) for d in os.listdir("data/cheetah") if d.endswith(".npy")])

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.future = future

        #make train-test-val split
        train_files = all_files[15:25]
        test_files = all_files[5:15]
        valid_files = all_files[:5]

        self.train_x, self.train_y = self._load_files(train_files)
        self.test_x, self.test_y = self._load_files(test_files)
        self.valid_x, self.valid_y = self._load_files(valid_files)
        self.feature_labels = [f"vector {i}" for i in range(self.train_x.shape[2])]
        super().__init__()

    def _load_files(self,files):
        all_x = []
        all_y = []
        for f in files:
           
            arr = np.load(f)
            arr = arr.astype(np.float32)
            x,y = cut_in_sequences(arr,seq_len=self.seq_len,inc=10,prognosis=self.future)

            all_x.extend(x)
            all_y.extend(y)

        return np.stack(all_x,axis=1),np.stack(all_y,axis=1)

class NeuronData(DataBaseClass):
    def __init__(self,binwidth=0.05,sigma=7,seq_len=32,future=1,iterative_forecast=False,batch_size=16,cross_val_fold=None,scheduled_sampling=False):
        self.seq_len = int(seq_len)
        self.batch_size = batch_size
        self.binwidth = binwidth
        self.future = future
        self.iterative_forecast = iterative_forecast
        self.sigma = sigma
        self.n_iterative_forecasts = 0
        self.scheduled_sampling = scheduled_sampling
        if self.iterative_forecast:
            self.n_iterative_forecasts = future
            self.future = 1
        self.n_forecasts = max(self.future,self.n_iterative_forecasts)
        """ 
        x: (Neurons,T) -> transpose to (T,Neurons)
        train_x after cut : (N_sequences) * (seq_len,Neurons)
        after stack (after make sequences):  (seq_len,N_sequences,T,Neurons)
        """
        super().__init__()
        self.load_data(cross_val_fold)
        self.set_kfold_splits()

    def load_data(self, cross_val_fold=None):
        self.cross_val_fold = cross_val_fold
        if self.binwidth == 0.05:
            x = np.load(f"data/neurons/activations_ADL1_2023-10-24_22-40-25=s{self.sigma}.npy")
        elif self.binwidth == 0.5:
            x = np.load("data/neurons/activations_f0.5_w1.0.npy")
        x = x.astype(np.float32)
        x = np.transpose(x)
        self.in_features = x.shape[-1]
        self.out_features = x.shape[-1]
        # train val test split
        self.valid_chunk,self.test_chunk, self.train_chunk = self.train_test_split(x)
        self.increment = max(int(x.shape[0] / 2000),2)

        self.make_sequences() 
        self.laserpoint_timestamps = None        
        self.feature_labels = [f"neuron {i}" for i in range(self.train_y.shape[2])]
        self.normalize_data()

    def train_test_split(self,x) :
        """Create train val test split. We need each of these splits to be created from sequential chunks
        but we also want to introduce some randomness, Therefore we cut 2 random chunks: a val and test chunk.
        """
        val_size = int(np.floor(x.shape[0] *0.15))
        test_size = int(np.floor(x.shape[0] *0.1))

        if self.cross_val_fold:
            _, test_indices= self.kfold_splits[self.cross_val_fold-1]
            _, val_indices = self.kfold_splits[(self.cross_val_fold-1+2)%5]
            start_val = val_indices[0]*32
            start_test = test_indices[0]*32
        else:
            # start of the validation chunk                         [all start positions in the sequence minus some leading space]
            start_val = np.random.RandomState(56034).randint(0, (self.n_bins - val_size))
            
            # pick start of test chunk from set of suitable start indices of test chunk. Such that val and test chunk do not overlap
            #                                    [all start positions in the sequence ]     -       [indices in val chunk and some leading space (prevent overlap between test and val)]
            test_start_indices = list(set(range((self.n_bins - (self.seq_len+self.future)) - test_size)) - set(range(start_val-test_size, start_val + val_size)))
            start_test = np.random.RandomState(49823).choice(test_start_indices)

        if not self.use_validation_data:  
            start_val = start_test  
            val_size = test_size

        # Extract the val and test chunk and assign the rest to the train chunk
        val_data = x[start_val:start_val +val_size]
        test_data = x[start_test:start_test +test_size]
        chunk_indices = np.sort(list(set(range(start_test,start_test+test_size)) | set(range(start_val,start_val+val_size))))

        #obtain the space between test and val chunks. we cannot do np indexing on x(~val_indices|test_indices) because x is a list of objects
        try:
            chunk_separation = np.where(chunk_indices[:-1] != (chunk_indices[1:] -1) )[0][0]
        except:
            chunk_separation = 0
        train_data = [
            x[:chunk_indices[0]],
            x[chunk_indices[chunk_separation]:chunk_indices[chunk_separation+1]],
            x[chunk_indices[-1]:]
        ]

        print(f"val idx: {start_val} - {start_val + val_size} ({val_size}) test idx: {start_test} {start_test + test_size} ({test_size})")
        return (val_data, test_data,train_data)  
    
class NeuronLaserData(DataBaseClass):
    def __init__(self,binwidth=0.05,sigma=7,seq_len=32,future=1,iterative_forecast=False,batch_size=16,cross_val_fold=None,scheduled_sampling =False):
        self.seq_len = int(seq_len)
        self.batch_size = batch_size
        self.binwidth = binwidth
        self.future = future
        self.iterative_forecast = iterative_forecast
        self.sigma = sigma
        self.scheduled_sampling = scheduled_sampling
        self.n_iterative_forecasts = 0
        if self.iterative_forecast:
            self.n_iterative_forecasts = future
            self.future = 1
        self.n_forecasts = max(self.future,self.n_iterative_forecasts)
        """ 
        x: (Neurons,T) -> transpose to (T,Neurons)
        train_x after cut : (N_sequences) * (seq_len,Neurons)
        after stack:  (seq_len,N_sequences,T,Neurons)
        after stack (after make sequences):  (seq_len,N_sequences,T,Neurons)
        """
        super().__init__()
        self.load_data(cross_val_fold)
        self.set_kfold_splits()

    def load_data(self, cross_val_fold=None):
        self.cross_val_fold = cross_val_fold
        if self.binwidth == 0.05:
            # x = np.load(f"data/neurons/activations_ADL1_2023-10-24_22-40-25_s{self.sigma}.npy")
            # x2 = np.load(f"data/neurons/laserpulses_ADL1_2023-10-24_22-40-25_s{self.sigma}.npy")
            x = np.load(f"data/neurons/activations_ADL1_2023-07-31_00-09-22_s{self.sigma}.npy")
            x2 = np.load(f"data/neurons/laserpulses_ADL1_2023-07-31_00-09-22_s{self.sigma}.npy")
        # elif self.binwidth == 0.5:
        #     x = np.load("data/neurons/activations_f0.5_w1.0.npy")
        #     x2 = np.load("data/neurons/laserpulses_f0.5_w1.0.npy")
        x = np.concatenate([x,x2], 0)
        x = x.astype(np.float32)
        x = np.transpose(x)
        self.in_features = x.shape[-1]
        self.out_features = x.shape[-1] - 1
        # train val test split
        self.n_bins = x.shape[0] 
        self.valid_chunk,self.test_chunk, self.train_chunk = self.train_test_split(x)
        
        self.increment = max(int(x.shape[0] / 2000),2)
        self.make_sequences()   
        print(self.future)
        self.laserpoint_timestamps = np.swapaxes(self.test_plot_x,0,1)[1:,-self.future:,-1].flatten()
        self.feature_labels = [f"vector {i}" for i in range(self.train_x.shape[2])]
        self.normalize_data()

    def train_test_split(self,x) :
        """Create train val test split. We need each of these splits to be created from sequential chunks
        but we also want to introduce some randomness, Therefore we cut 2 random chunks: a val and test chunk.
        """
        val_size = int(np.floor(x.shape[0] *0.15))
        test_size = int(np.floor(x.shape[0] *0.1))

        if self.cross_val_fold:
            _, test_indices= self.kfold_splits[self.cross_val_fold-1]
            _, val_indices =self.kfold_splits[(self.cross_val_fold-1+2)%5]
            start_val = val_indices[0]*32
            start_test = test_indices[0]*32
        else:
            # start of the validation chunk                         [all start positions in the sequence minus some leading space]
            start_val = np.random.RandomState(56034).randint(0, (self.n_bins - val_size))
            
            # pick start of test chunk from set of suitable start indices of test chunk. Such that val and test chunk do not overlap
            #                                    [all start positions in the sequence ]     -       [indices in val chunk and some leading space (prevent overlap between test and val)]
            test_start_indices = list(set(range((self.n_bins - (self.seq_len+self.future)) - test_size)) - set(range(start_val-test_size, start_val + val_size)))
            start_test = np.random.RandomState(49823).choice(test_start_indices)
        if not self.use_validation_data:  
            start_val = start_test  
            val_size = test_size
        # Extract the val and test chunk and assign the rest to the train chunk
        val_data = x[start_val:start_val +val_size]
        test_data = x[start_test:start_test +test_size]
        chunk_indices = np.sort(list(set(range(start_test,start_test+test_size)) | set(range(start_val,start_val+val_size))))

        #obtain the space between test and val chunks. we cannot do np indexing on x(~val_indices|test_indices) because x is a list of objects
        try:
            chunk_separation = np.where(chunk_indices[:-1] != (chunk_indices[1:] -1) )[0][0]
            train_data = [
                x[:chunk_indices[0]],
                x[chunk_indices[chunk_separation]:chunk_indices[chunk_separation+1]],
                x[chunk_indices[-1]:]
            ]
        except:
            train_data = [
                x[:chunk_indices[0]],
                x[chunk_indices[-1]:]
            ]

        print(f"val idx: {start_val} - {start_val + val_size} ({val_size}) test idx: {start_test} {start_test + test_size} ({test_size})")
        return (val_data, test_data,train_data)  
    
class ForecastModel:    
    def __init__(self,model_id = None,_data=None,task=None,checkpoint_id = None):
        self.task = task
        self.set_model_id(model_id,checkpoint_id)
        self.set_data(_data)

    def set_model_id(self,model_id,checkpoint_id):
        self.model_id = model_id
        self.experiment_name = f"{self.task}_{self.model_id}"
        self.store_dir = f"./checkpoints/{self.task}_{self.model_id}"
        self.load_dir =  f"./checkpoints/{self.task}_{checkpoint_id}" if checkpoint_id else self.store_dir 
        self.load_path = f"{self.load_dir}/last.ckpt"  

    def set_data(self,_data):
        self.future = _data.future
        self.n_iterative_forecasts = _data.n_iterative_forecasts
        self.n_forecasts = _data.n_forecasts
        self._loaderfunc = _data.get_dataloader
        self.feature_labels = _data.feature_labels
        self.in_features = _data.in_features
        self.out_features = _data.out_features
        self.mean = _data.mean
        self.std = _data.std
        self.seq_len = _data.seq_len
        self._loaderfunc = _data.get_dataloader
        self.batch_size = _data.batch_size
        self.iterative_forecast = _data.iterative_forecast
        self.data_loader = _data.get_dataloader
        self.cross_val_fold = _data.cross_val_fold
        self.scheduled_sampling = _data.scheduled_sampling
        self.test_laser_timestaps = _data.laserpoint_timestamps

    def set_model(self):
        if(self.model_type.startswith("ltc")):
            # wiring = AutoNCP(model_size,out_features)
            wiring = FullyConnected(self.model_size,self.out_features)
            self._model = LTC(self.in_features,wiring,batch_first=True,mixed_memory=self.mixed_memory)

        # lr_logger = LearningRateMonitor(logging_interval='step') # this is a callback

    def set_optuna_trainer(self):
        self.trainer = pl.Trainer(
            logger = False,
            max_epochs= self.epochs,
            gradient_clip_val=1,
            accelerator= 'cpu' if self.gpus is None else 'gpu',
            default_root_dir = self.store_dir,
            devices=self.gpus,
            callbacks=[PyTorchLightningPruningCallback(self.trial, monitor="val_loss")],
            enable_checkpointing=False
        )

    def set_fit_trainer(self):
        # Modelcheckoint is used to obtain the last and best checkpoint
        # dirpath should be load dir, which should refer to checkpoint 
        # 
        self.checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath=self.store_dir ,
            filename="best",
            save_on_train_epoch_end=True,
            save_last=True
        )       

        self.tensorboard_logger = TensorBoardLogger(save_dir="log",
                version = f"{self.model_id}",
                name=f"{self.task}_{self.model_type}",
                default_hp_metric=False)

        self.trainer = pl.Trainer(
            logger = self.tensorboard_logger,
            max_epochs= self.epochs,
            gradient_clip_val=1,
            accelerator= 'cpu' if self.gpus is None else 'gpu',
            devices=self.gpus,
            callbacks=[self.checkpoint_callback],
        )

    def fit(self,trial=None,epochs=100,learning_rate=1e-2,cosine_lr=False,model_size =None,mixed_memory=True,model_type="ltc",gpus=None,future_loss = False,reset = False):
            self.epochs = epochs
            self.cosine_lr = cosine_lr   
            self.gpus = gpus
            self.trial = trial
            self.model_size = int(model_size)
            self.mixed_memory = mixed_memory
            self.model_type = model_type
            self.learning_rate = float(learning_rate)
            if future_loss:
                self.loss = MSELossfuture(n_future=self.future)
            else:
                self.loss = torch.nn.MSELoss()
            self.set_model()

            if not trial:
                self.set_fit_trainer()
            else:
                self.set_optuna_trainer()
            if self.scheduled_sampling:

                self.learn = ScheduledSamplingSequenceLearner(model= self._model,loss_func = self.loss,lr=self.learning_rate,cosine_lr=self.cosine_lr,
                                            iterative_forecast=self.iterative_forecast,_loaderfunc=self._loaderfunc, n_iterative_forecasts = self.n_iterative_forecasts, 
                                            n_iterations=(self.batch_size * self.epochs),future = self.future,denormalize = self.denormalize)                
            else:
                self.learn = SequenceLearner(model= self._model,loss_func = self.loss,lr=self.learning_rate,cosine_lr=self.cosine_lr,
                                            iterative_forecast=self.iterative_forecast,_loaderfunc=self._loaderfunc, n_iterative_forecasts = self.n_iterative_forecasts, 
                                            n_iterations=(self.batch_size * self.epochs),future = self.future,denormalize = self.denormalize)
            try:
                if not reset and not trial:
                    print("training from loaded ckpt")
                    self.trainer.fit(self.learn,ckpt_path=self.load_path)
                else:
                    raise ValueError
            except Exception as e:
                print(e)
                if hasattr(self,"tensorboard_logger"):
                    self.tensorboard_logger.log_hyperparams({
                        "lr": learning_rate,
                        "cosine_lr": cosine_lr,
                        "seq_len":self.seq_len,
                        "future": self.n_forecasts,
                        "model_size": self.model_size,
                        "future loss": future_loss,
                        "scheduled sampling": self.scheduled_sampling
                    })
                self.trainer.fit(self.learn)

        
            if trial:
               return self.trainer.callback_metrics["val_loss"].item()
            else:
                try:
                    test_score = self.trainer.test(self.learn, ckpt_path=f"{self.store_dir}/best.ckpt")
                except Exception as e:
                    print(e)
                    test_score = self.trainer.test(self.learn, ckpt_path=f"{self.load_dir}/best.ckpt")
                return test_score

    def denormalize(self,tensor,values="x"):
        # return tensor * self.std[values] + self.mean[values]
        #because if we use realtime data we do not have the laser-activation data channel in the input x data so this is not normalized
        return tensor * self.std[values][...,:tensor.shape[-1]] + self.mean[values][...,:tensor.shape[-1]]

    def normalize(self,tensor,values="x"):
        # return (tensor - self.mean[values]) / self.std[values]
        #because if we use realtime data we do not have the laser-activation data channel in the input x data so this is not normalized
        return (tensor - self.mean[values][...,:tensor.shape[-1]]) / self.std[values][...,:tensor.shape[-1]]

    def test(self,iterative_forecast=False,checkpoint="best"):
        # try:
        #     test_score = self.trainer.test(self.learn, ckpt_path=f"{self.store_dir}/{checkpoint}.ckpt")
        # except Exception as e:
        #     print(e)
        #     test_score = self.trainer.test(self.learn, ckpt_path=f"{self.load_dir}/{checkpoint}.ckpt")
        self.plot_test(version=checkpoint,iterative_forecast=iterative_forecast)

        # return test_score
    
    def plot_test(self,version="before",iterative_forecast=False):
        predictions = self.trainer.predict(dataloaders = self._loaderfunc(subset="test_plot"),ckpt_path=f"{self.load_dir}/{version}.ckpt")
        y,y_hat,error = map(torch.cat, zip(*predictions))
        if self.in_features >self.out_features:
            laser_timestamps = self.test_laser_timestaps
        else: 
            laser_timestamps = []
        vlines = laser_timestamps.nonzero()[0]

        if self.n_forecasts> 1:
            fig, axes = plt.subplots(self.out_features, 1, figsize=(30,4*self.out_features),constrained_layout=True)
            axes = axes.ravel()
            for (feat_nr,ax) in zip(range(self.out_features),axes):
                ax.plot(y[:,feat_nr],label="Target output",linewidth=1)
                ax.plot(y_hat[:,feat_nr], label="NCP output",linewidth=1)
                ax.set_title(f"{self.feature_labels[feat_nr]}",loc = 'left')
                ax.legend(loc='upper right')
                ax.vlines(vlines,0,max(y[:,feat_nr]))
            
            plt.suptitle(f"{version} training")
            plt.savefig(f"results/{self.task}/{self.model_id}_{version}_all_predictions.jpg")
            plt.close()


            fig, axes = plt.subplots(self.out_features, 1, figsize=(30,4*self.out_features),constrained_layout=True)
            axes = axes.ravel()
            for (feat_nr,ax) in zip(range(self.out_features),axes):
                ax.plot(error[:,feat_nr], label="Prediction error",linewidth=1)
                ax.set_title(f"{self.feature_labels[feat_nr]}",loc = 'left')
                ax.legend(loc='upper right')
                ax.vlines(vlines,0,max(y[:,feat_nr]))

            plt.suptitle(f"{version} training")
            plt.savefig(f"results/{self.task}/{self.model_id}_{version}_all_predictions-error.jpg")
            plt.close()

        pool = multiprocessing.Pool(processes=self.n_forecasts)
        inputs = list([(future_point,y,y_hat,error,self.n_forecasts,version,self.out_features,self.feature_labels,self.task,self.model_id,self.test_laser_timestaps) for future_point in range(self.n_forecasts)])
        pool.starmap(plot_future_point,inputs)
        return

    def analyse(self):
        predictions = self.trainer.predict(dataloaders = self._loaderfunc(subset="test_plot"),ckpt_path=f"{self.load_dir}/{version}.ckpt")
        y,y_hat,error = map(torch.cat, zip(*predictions))


        future_point_indices = torch.LongTensor(list(range(future_point,y.shape[0],n_forecasts)))
        y_for_future = np.take(y, future_point_indices, 0)
        y_hat_for_future = np.take(y_hat,future_point_indices, 0)
        error_for_future = np.take(error, future_point_indices, 0)
        laser_timestamps_for_future = np.take(laser_timestamps, future_point_indices[future_point_indices<laser_timestamps.shape[0]], 0)
        vlines = laser_timestamps_for_future.nonzero()[0]


        # plot the predictions for each feature in a subplot
        fig, axes = plt.subplots(out_features, 1, figsize=(30,4*out_features),constrained_layout=True)
        axes = axes.ravel()
        for (feat_nr,ax) in zip(range(out_features),axes):
            ax.plot(y_for_future[:,feat_nr],label="Target output",linewidth=1)
            ax.plot(y_hat_for_future[:,feat_nr], label="NCP output",linewidth=1)
            ax.set_title(f"{feature_labels[feat_nr]}",loc = 'left')
            ax.legend(loc='upper right')
            ax.vlines(vlines,0,max(y_for_future[:,feat_nr]))

        plt.suptitle(f"{version} training")
        plt.savefig(f"results/{task}/{model_id}_{version}_{future_point}.jpg")
        plt.close()

        # plot the error for each feature in a subplot
        fig, axes = plt.subplots(out_features, 1, figsize=(30,4*out_features),constrained_layout=True)
        axes = axes.ravel()
        for (feat_nr,ax) in zip(range(out_features),axes):
            error_magnitude = (error_for_future[:,feat_nr]**2).mean()
            ax.plot(error_for_future[:,feat_nr], label="Prediction error",linewidth=1)
            ax.set_title(f"{feature_labels[feat_nr]} MSE {error_magnitude:.4f}",loc = 'left')
            ax.legend(loc='upper right')
            ax.vlines(vlines,0,max(y_for_future[:,feat_nr]))

        plt.suptitle(f"{version} training")
        plt.savefig(f"results/{task}/{model_id}_{version}_{future_point}-error.jpg")
        plt.close()
        print(f"saved plots in results/{task}/{model_id}_{version}")

def plot_future_point(future_point,y,y_hat,error,n_forecasts,version,out_features,feature_labels,task,model_id,laser_timestamps):
    future_point_indices = torch.LongTensor(list(range(future_point,y.shape[0],n_forecasts)))
    y_for_future = np.take(y, future_point_indices, 0)
    y_hat_for_future = np.take(y_hat,future_point_indices, 0)
    error_for_future = np.take(error, future_point_indices, 0)
    laser_timestamps_for_future = np.take(laser_timestamps, future_point_indices[future_point_indices<laser_timestamps.shape[0]], 0)
    vlines = laser_timestamps_for_future.nonzero()[0]


    # plot the predictions for each feature in a subplot
    fig, axes = plt.subplots(out_features, 1, figsize=(30,4*out_features),constrained_layout=True)
    axes = axes.ravel()
    for (feat_nr,ax) in zip(range(out_features),axes):
        ax.plot(y_for_future[:,feat_nr],label="Target output",linewidth=1)
        ax.plot(y_hat_for_future[:,feat_nr], label="NCP output",linewidth=1)
        ax.set_title(f"{feature_labels[feat_nr]}",loc = 'left')
        ax.legend(loc='upper right')
        ax.vlines(vlines,0,max(y_for_future[:,feat_nr]))

    plt.suptitle(f"{version} training")
    plt.savefig(f"results/{task}/{model_id}_{version}_{future_point}.jpg")
    plt.close()

    # plot the error for each feature in a subplot
    fig, axes = plt.subplots(out_features, 1, figsize=(30,4*out_features),constrained_layout=True)
    axes = axes.ravel()
    for (feat_nr,ax) in zip(range(out_features),axes):
        error_magnitude = (error_for_future[:,feat_nr]**2).mean()
        ax.plot(error_for_future[:,feat_nr], label="Prediction error",linewidth=1)
        ax.set_title(f"{feature_labels[feat_nr]} MSE {error_magnitude:.4f}",loc = 'left')
        ax.legend(loc='upper right')
        ax.vlines(vlines,0,max(y_for_future[:,feat_nr]))

    plt.suptitle(f"{version} training")
    plt.savefig(f"results/{task}/{model_id}_{version}_{future_point}-error.jpg")
    plt.close()
    print(f"saved plots in results/{task}/{model_id}_{version}")

class RealtimeNeuronLaserData(NeuronLaserData):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.pipe_connection = None

    def prepare_realtime_data(self,_path):
        x = np.load(_path)
        x = x.astype(np.float32)
        x = np.transpose(x)
        
        setattr(self,"predict_x",x)   
        # print("sending ",flush=True)
        in_x, in_x_de = self.get_dataloader(subset="predict")
        self.pipe_connection.send((in_x, in_x_de))
    
    def set_pipe(self, pipe_connection):
        self.pipe_connection = pipe_connection

class RealtimeForecastModel(ForecastModel):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.total_read_timesteps = 0
        self.trainer = pl.Trainer()
        self.prepare_realtime_data = kwargs["_data"].prepare_realtime_data
        self.norm_recorded_history = torch.empty((1,0,self.in_features-1),dtype=torch.float32)
        self.recorded_history = torch.empty((0,self.in_features-1),dtype=torch.float32)
        self.predict_pos_history = torch.full((self.n_forecasts,self.in_features-1),0)
        self.predict_neg_history = torch.full((self.n_forecasts,self.in_features-1),0)


    def run(self,recording_id,gpus,pipe_connection,plot_sender,model_size,mixed_memory,model_type):  
        self.model_size = model_size
        self.mixed_memory = mixed_memory
        self.model_type = model_type

            
        self.set_model()
        try:    
            self.learn = SequenceLearner.load_from_checkpoint(f"{self.load_dir}/best.ckpt",model=self._model,map_location=torch.device(gpus))
        except: 
            self.learn = SequenceLearner.load_from_checkpoint(f"{self.load_dir}/best.ckpt",model=self._model,map_location=torch.device(gpus))
        self.learn.eval()
        self.pipe_connection = pipe_connection
        self.plot_sender = plot_sender
        self.recording_id = recording_id

        self.predict()



    def predict(self):
        self.device = next(self._model.parameters()).device
        # while not self.stop_event.is_set():
        self.norm_recorded_history = self.norm_recorded_history.to(self.device,non_blocking=True)
        print("start predict")
        pos_laser_norm_value = 1 - self.mean["x"][...,-1] / self.std["x"][...,-1]
        neg_laser_norm_value = 0 - self.mean["x"][...,-1] / self.std["x"][...,-1]
        laser_feature = torch.full((1,32,1))
        while True:
            _start = dt.datetime.now()
            if  not self.pipe_connection.poll():
                continue
            in_x, in_x_de = self.pipe_connection.recv()
            in_x = in_x.to(self.device,non_blocking=True) # shape of (seq_len,1,neurons) > (1,seq_len,neurons) 
            new_read_timesteps = in_x.shape[1]
            self.total_read_timesteps += new_read_timesteps
            """Please check the following. why is in_x_de indexed by new_read_timesteps? check get_dataloader("predict") in DataBaseClass """
            self.norm_recorded_history = torch.concat((self.norm_recorded_history,in_x), dim= 1) 
            self.recorded_history = torch.concat((self.recorded_history,torch.tensor(in_x_de[-new_read_timesteps:])), dim= 0) 

            if self.total_read_timesteps < self.seq_len:
                print("too small",self.norm_recorded_history.shape,flush=True)
                self.predict_pos_history = torch.concat((self.predict_pos_history,torch.full((new_read_timesteps,self.in_features-1),0)), dim= 0) 
                self.predict_neg_history = torch.concat((self.predict_neg_history,torch.full((new_read_timesteps,self.in_features-1),0)), dim= 0) 
                continue

            missed_predictions = max(0,new_read_timesteps - self.n_forecasts)
            in_x = self.norm_recorded_history[:,-self.seq_len:,:]
            x_pos = x_neg = in_x # (1,seq_len,neurons)
            y_hat_pos = y_hat_neg  = torch.empty((1,self.n_iterative_forecasts,in_x.shape[-1]), device=self.device)
            for i in range(self.n_iterative_forecasts):
                """we only add a 1 for laser activation at the 2 bins with most lag, no other bins"""
                x_pos_in = torch.cat((x_pos,torch.full((1,32,1),pos_laser_norm_value if i > 3 else neg_laser_norm_value,device=self.device)),dim=-1)
                x_neg_in = torch.cat((x_neg,torch.full((1,32,1), 0,device=self.device)),dim=-1)
                x = torch.cat((x_pos_in,x_neg_in),dim=0)
                pred, _ = self._model.forward(x) # (2, last timestep, neurons(excluding activation))
                (next_step_pos,next_step_neg) = pred
                y_hat_pos[:,i] = next_step_pos[-1:,:]
                y_hat_neg[:,i] = next_step_neg[-1:,:]
                x_pos = torch.cat((x_pos[:,1:],next_step_pos[-1:,:].unsqueeze(0)),dim=1)
                x_neg = torch.cat((x_neg[:,1:],next_step_neg[-1:,:].unsqueeze(0)),dim=1) 
            """TODO Now we compare the pos (stimulation) predictions and neg (absence) predictions"""
            """Now we plot the recorded history of the signal followed by the forecasts
                    for every neuron + activation
                    make sure to include the laseractivation feature in the predictions
            """
            print('Duration: {:.4f}'.format((dt.datetime.now() - _start).total_seconds() *1000),flush=True)
            y_hat_pos_de = self.denormalize(y_hat_pos.detach().cpu(),"y").flatten(0,1) # (5,17)
            y_hat_neg_de = self.denormalize(y_hat_neg.detach().cpu(),"y").flatten(0,1)
            self.predict_pos_history = torch.concat((self.predict_pos_history,torch.full((missed_predictions,self.in_features-1),0),y_hat_pos_de), dim= 0) 
            self.predict_neg_history = torch.concat((self.predict_neg_history,torch.full((missed_predictions,self.in_features-1),0),y_hat_neg_de), dim= 0) 
            plot_length = min(100, self.total_read_timesteps)
            self.plot_sender.send((plot_length,
                    self.predict_pos_history[-(plot_length+self.n_forecasts):],self.predict_neg_history[-(plot_length+self.n_forecasts):], 
                    self.recorded_history[-plot_length:]))
