import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import sys
import argparse
import datetime as dt
from datetime import timedelta
import time
import multiprocessing
from functools import partial

from ncps.torch import LTC
from ncps.wirings import AutoNCP, FullyConnected


from LTC_learner import SequenceLearner
from load_data import load_training_data

import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint


import optuna
from optuna.integration import PyTorchLightningPruningCallback

from concurrent.futures import ThreadPoolExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


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

class DataBaseClass:
    def __init__(self):
        self.load_data()
        dataset_names = ["train_x","train_y","valid_x", "valid_y","test_x","test_y","test_plot_x","test_plot_y"]
        train_with_noise = False
        for _name in dataset_names:
            _array = getattr(self,_name)
            print(f"{_name} shape: ", str(_array.shape))
            _array = self.normalize(_array,_name.split("_")[-1])
            print(f"{_name} shape: ", str(_array.shape),f"{_name} post normalisation: ",_array.mean())
            if "train" in _name and train_with_noise:
                noise = np.random.normal(0, 0.1, _array.shape) 
                _array = _array + noise
                setattr(self,_name, _array)
                print(f"{_name} post noise: ",str(_array.mean()))              

        self.in_features = self.train_x.shape[2]
        self.out_features = self.train_y.shape[2]

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
        x_data = torch.permute(torch.tensor(getattr(self,f"{subset}_x")),(1,0,2)) # _batch = (B_size, Timesteps, Features)
        y_data = torch.permute(torch.tensor(getattr(self,f"{subset}_y")),(1,0,2))
        return data.DataLoader(
            data.TensorDataset(x_data, y_data),
            batch_size =  self.n_forecasts if subset == "test_plot" and self.iterative_forecast else self.batch_size,
            shuffle=True if subset == "train" else False,
            num_workers = 1 if subset == "test_plot" else 4,
        )

    def normalize(self, tensor,values="x"):
        tensor = torch.tensor(tensor)
        if not hasattr(self,"std") or not hasattr(self,"mean"):
            self.mean = {}
            self.std = {}
        if not values in self.mean or not values in self.std:
            self.mean[values] = torch.mean(tensor, dim=(0, 1), keepdim=True)
            self.std[values] = torch.std(tensor, dim=(0, 1), keepdim=True)
            res = self.std[values].clone()
            res[self.std[values]==0] = torch.tensor(1)
            self.std[values] = res
            del res
        return (tensor - self.mean[values][...,:tensor.shape[-1]]) / self.std[values][...,:tensor.shape[-1]]

    @classmethod    
    def denormalize(self,tensor,values="x"):
        return tensor * self.std[values][...,:tensor.shape[-1]] + self.mean[values][...,:tensor.shape[-1]]



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
    def __init__(self,binwidth=0.05,seq_len=32,future=1,iterative_forecast=False,batch_size=16):

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.future = future
        self.iterative_forecast = iterative_forecast
        self.n_iterative_forecasts = 0
        if self.iterative_forecast:
            self.n_iterative_forecasts = future
            self.future = 1
        """ 
        x: (Neurons,T) -> transpose to (T,Neurons)
        train_x after cut : (N_sequences) * (seq_len,Neurons)
        after stack (after make sequences):  (seq_len,N_sequences,T,Neurons)
        """
        if binwidth == 0.05:
            x = np.load("data/neurons/activations_ADL1_2023-10-24_22-40-25.npy")
        elif binwidth == 0.5:
            x = np.load("data/neurons/activations_f0.5_w1.0.npy")
        x = x.astype(np.float32)
        x = np.transpose(x)

        # train val test split
        self.valid_chunk,self.test_chunk, self.train_chunk = self.train_test_split(x)
        self.increment = max(int(x.shape[0] / 1000),2)
        self.make_sequences()         
        self.feature_labels = self.feature_labels = [f"neuron {i}" for i in range(self.train_y.shape[2])]
        super().__init__()

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
                    temp_x.extend(temp_x_arr)
                    temp_y.extend(temp_y_arr)
            else:
                temp_x,temp_y = cut_in_sequences(_array,seq_len=self.seq_len, inc=self.increment if not _split == "test" else 1,prognosis=self.future)
            temp_x = np.stack(temp_x, axis=1) 
            temp_y = np.stack(temp_y, axis=1)
            setattr(self,f"{_split}_x",temp_x)    
            setattr(self,f"{_split}_y",temp_y)
            if _split == "test" :
                # in case of iterative forecast: prognosis=1. per batch of n_iterative we will use the model predictions as input
                # otherwise: prognosis = future. the data to plot will be composed of the predicted items. (hence the increment)
                temp_x,temp_y = cut_in_sequences(_array,seq_len=self.seq_len, inc=self.future,prognosis=self.future)
                temp_x = np.stack(temp_x, axis=1) 
                temp_y = np.stack(temp_y, axis=1)
                setattr(self,f"predict_x",temp_x)    
                setattr(self,f"predict_y",temp_y)  

    def train_test_split(self,x):
        """Create train val test split. We need each of these splits to be created from sequential chunks
        but we also want to introduce some randomness, Therefore we cut 2 random chunks: a val and test chunk.
        """
        val_size = int(np.floor(x.shape[0] *0.15))
        test_size = int(np.floor(x.shape[0] *0.10))
        N = x.shape[0]  
        # start of the validation chunk                         [all start positions in the sequence minus some leading space]
        start_val = np.random.RandomState(56034).randint(0, (N - val_size))
        
        # pick start of test chunk from set of suitable start indices of test chunk. Such that val and test chunk do not overlap
        #                                    [all start positions in the sequence ]     -       [indices in val chunk and some leading space (prevent overlap between test and val)]
        test_start_indices = list(set(range((N - (self.seq_len+self.future)) - test_size)) - set(range(start_val-test_size, start_val + val_size)))
        start_test = np.random.RandomState(49823).choice(test_start_indices)
        
        # Extract the val and test chunk and assign the rest to the train chunk
        val_data = x[start_val:start_val +val_size]
        test_data = x[start_test:start_test +test_size]
        chunk_indices = np.sort(list(set(range(start_test,start_test+test_size)) | set(range(start_val,start_val+val_size))))
        try:
            chunk_separation = np.where(chunk_indices[:-1] != (chunk_indices[1:] -1) )[0][0]
        except:
            chunk_separation = 0
        train_data = [
            x[:chunk_indices[0]],
            x[chunk_indices[chunk_separation]:chunk_indices[chunk_separation+1]],
            x[chunk_indices[-1]:]
        ]


        return (val_data, test_data,train_data)  

class NeuronLaserData(DataBaseClass):
    def __init__(self,binwidth=0.05,seq_len=32,future=1,iterative_forecast=False,batch_size=16):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.binwidth = binwidth
        self.future = future
        self.iterative_forecast = iterative_forecast
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

    def load_data(self):
        if self.binwidth == 0.05:
            x = np.load("data/neurons/activations_ADL1_2023-10-24_22-40-25.npy")
            x2 = np.load("data/neurons/laserpulses_ADL1_2023-10-24_22-40-25.npy")
        elif self.binwidth == 0.5:
            x = np.load("data/neurons/activations_f0.5_w1.0.npy")
            x2 = np.load("data/neurons/laserpulses_f0.5_w1.0.npy")
        x = np.concatenate([x,x2], 0)
        x = x.astype(np.float32)
        x = np.transpose(x)
        # train val test split
        self.valid_chunk,self.test_chunk, self.train_chunk = self.train_test_split(x)
        self.increment = max(int(x.shape[0] / 1000),2)
        self.make_sequences()   
        self.feature_labels = [f"vector {i}" for i in range(self.train_x.shape[2])]

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
            temp_y = np.stack(temp_y, axis=1)[:,:,:-1]
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
                temp_y = np.stack(temp_y, axis=1)[:,:,:-1]
                setattr(self,f"test_plot_x",temp_x)    
                setattr(self,f"test_plot_y",temp_y)  

    def train_test_split(self,x) :
        """Create train val test split. We need each of these splits to be created from sequential chunks
        but we also want to introduce some randomness, Therefore we cut 2 random chunks: a val and test chunk.
        """
        val_size = int(np.floor(x.shape[0] *0.15))
        test_size = int(np.floor(x.shape[0] *0.1))
        N = x.shape[0]  
        # start of the validation chunk                         [all start positions in the sequence minus some leading space]
        start_val = np.random.RandomState(56034).randint(0, (N - val_size))
        
        # pick start of test chunk from set of suitable start indices of test chunk. Such that val and test chunk do not overlap
        #                                    [all start positions in the sequence ]     -       [indices in val chunk and some leading space (prevent overlap between test and val)]
        test_start_indices = list(set(range((N - (self.seq_len+self.future)) - test_size)) - set(range(start_val-test_size, start_val + val_size)))
        start_test = np.random.RandomState(49823).choice(test_start_indices)
        
        # Extract the val and test chunk and assign the rest to the train chunk
        val_data = x[start_val:start_val +val_size]
        test_data = x[start_test:start_test +test_size]
        chunk_indices = np.sort(list(set(range(start_test,start_test+test_size)) | set(range(start_val,start_val+val_size))))
        try:
            chunk_separation = np.where(chunk_indices[:-1] != (chunk_indices[1:] -1) )[0][0]
        except:
            chunk_separation = 0
        train_data = [
            x[:chunk_indices[0]],
            x[chunk_indices[chunk_separation]:chunk_indices[chunk_separation+1]],
            x[chunk_indices[-1]:]
        ]
        return (val_data, test_data,train_data)  
    

class ForecastModel:    
    def __init__(self,model_id = None,_data=None,task=None,model_size =None,mixed_memory=True,model_type="ltc",checkpoint_id = None):
        self.model_type = model_type
        self.task = task
        self.model_id = model_id
        self.model_size = model_size
        self.mixed_memory = mixed_memory
        self.experiment_name = f"{self.task}_{self.model_id}"
        self.store_dir = f"./checkpoints/{self.task}_{self.model_id}"
        self.load_dir =  f"./checkpoints/{self.task}_{checkpoint_id}" if checkpoint_id else self.store_dir 
        self.load_path = f"{self.load_dir}/last.ckpt"  
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

    def set_model(self):
        if(self.model_type.startswith("ltc")):
            # wiring = AutoNCP(model_size,out_features)
            wiring = FullyConnected(self.model_size,self.out_features)
            self._model = LTC(self.in_features,wiring,batch_first=True,mixed_memory=self.mixed_memory)

        # lr_logger = LearningRateMonitor(logging_interval='step') # this is a callback

    def fit(self,trial=None,_data=None,epochs=100,learning_rate=1e-2,cosine_lr=False,optimise=False,gpus=None,future_loss = False,reset = False):
            self.n_epochs = epochs
            print(self.load_dir,self.store_dir)
            if future_loss:
                loss = MSELossfuture(n_future=self.future)
            else:
                loss = torch.nn.MSELoss()

            if not optimise:
                self.set_model()
                # Modelcheckoint is used to obtain the last and best checkpoint
                # dirpath should be load dir, which should refer to checkpoint 
                # 
                self.checkpoint_callback = ModelCheckpoint(
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    dirpath=self.load_dir ,
                    filename="best",
                    save_on_train_epoch_end=True,
                    save_last=True
                )       

                tensorboard_logger = TensorBoardLogger(save_dir="log",
                        version = f"{self.model_id}",
                        name=f"{self.task}_{self.model_type}",
                        default_hp_metric=False)

                self.trainer = pl.Trainer(
                    logger = tensorboard_logger,
                    max_epochs= epochs,
                    gradient_clip_val=1,
                    accelerator= 'cpu' if gpus is None else 'gpu',
                    devices=gpus,
                    callbacks=[self.checkpoint_callback],
                )

                self.learn = SequenceLearner(self._model,loss,learning_rate=learning_rate,cosine_lr=cosine_lr,iterative_forecast=self.iterative_forecast,
                    _loaderfunc=self._loaderfunc,n_iterations=(self.batch_size * epochs))

            else:
                # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
                learning_rate = trial.suggest_float("learning_rate",1e-4,1e-2)
                cosine_lr = trial.suggest_int("cosine_lr",0,1) if cosine_lr else 0
                model_size = trial.suggest_int("model_size",self.out_features +3,48,step=4)

                self.model_size = model_size
                self.set_model()

                self.trainer = pl.Trainer(
                    logger = False,
                    max_epochs= epochs,
                    gradient_clip_val=1,
                    accelerator= 'cpu' if gpus is None else 'gpu',
                    devices=gpus,
                    callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],

                )

                self.learn = SequenceLearner(self._model,loss,learning_rate=learning_rate,cosine_lr=cosine_lr,
                    _loaderfunc=_data.get_dataloader,n_iterations=(_data.batch_size * epochs))
            if not reset:

                try:
                    self.trainer.fit(self.learn,ckpt_path=self.load_path)
                except:
                    tensorboard_logger.log_hyperparams({
                        "lr": learning_rate,
                        "cosine_lr": cosine_lr,
                        "seq_len":self.seq_len,
                        "future": self.n_forecasts,
                        "model_size": self.model_size,
                        "future loss": future_loss
                    })
                    self.trainer.fit(self.learn)
            else:
                tensorboard_logger.log_hyperparams({
                    "lr": learning_rate,
                    "cosine_lr": cosine_lr,
                    "seq_len":self.seq_len,
                    "future": self.n_forecasts,
                    "model_size": self.model_size,
                    "future loss": future_loss
                })
                self.trainer.fit(self.learn)
        
            if optimise:
               return self.trainer.callback_metrics["val_loss"].item()
            
    def denormalize(self,tensor,values="x"):
        return tensor * self.std[values] + self.mean[values]

    def normalize(self,tensor,values="x"):
        return (tensor - self.mean[values]) / self.std[values]
    
    def test(self,iterative_forecast=False,checkpoint="best"):
        # self.trainer.save_checkpoint(f"{self.checkpoint_dir}/last_epoch.ckpt")
        # self.learn.load_from_checkpoint(self.checkpoint_callback.best_model_path) 
        self.trainer.test(self.learn, ckpt_path=checkpoint)

        if self.n_epochs > 1: 
            self.plot_test(version=checkpoint,iterative_forecast=iterative_forecast)
        else:
            self.plot_test(version="before",iterative_forecast=iterative_forecast)

    def get_test_plot_data(self,iterative_forecast):
        torch.set_grad_enabled(False)
        self._model.eval()        
        y_list = []
        y_hat_list = []
        error_list = []
        for _batch in tqdm(self._loaderfunc(subset="test_plot"),position=0,leave=True):
            """" We either predict with iterative forecasting or not.
            with iterative forecasting we use the models prediction as input for the next timestep prediction.
                    every batch contains n_iterative_forecasts sequences.
            without iterative forecasting we predict 1 or more timsteps ahead based on 1 prediction """
            x, y = _batch                 # _batch = (B_size, Timesteps, Features) 
            # from y we take out the futures of each sequence
            # the shape of y is always (nsequences, seq len, features)
            # in case of not iterative we just select as below, resulting in size (16,5,17)
            # in case of iterative we do the same, resulting in (5,1,17)
            if iterative_forecast:
                y_hat = torch.zeros((1,y.shape[0],y.shape[2]), device=x.device) #(1, futures, neurons)
                y = y[:,-self.future:,:].transpose(0,1)
                n_predictions = y.shape[1]
                x = x[:1]  # first sequence of (\timesteps) length
                activation_status = int(x[0,-1,-1])
                """ forecast by recursive model calls"""
                for i in range(n_predictions):
                    next_step, _ = self._model.forward(x)  # (1, Timesteps, neurons) 
                    # we assume we deal with laser activity data which is part of x but not part of y 
                    y_hat[:,i] = next_step[:,-1:,:] 
                    next_step_x = torch.cat((next_step[:, -1:, :],torch.full((1,1,1),activation_status if i==0 else 0,device=x.device)),dim=-1)
                    x = torch.cat((x[:, 1:, :], next_step_x), dim=1) #input for next prediction
            else :
                y = y[:,-self.future:,:]
                """forecast directly multiple steps ahead with 1 model call"""
                y_hat, _ = self._model.forward(x) # (B_size, Timesteps, Features)
                y_hat = y_hat[:,-self.future:,:] #(sequences,timesteps,neurons) > (sequences,futures,neurons)

            y_de = self.denormalize(y,"y").flatten(0,1)
            y_hat_de = self.denormalize(y_hat,"y").flatten(0,1)
            error = (y_de - y_hat_de)
            y_list.append(y_de.detach().cpu())
            y_hat_list.append(y_hat_de.detach().cpu())
            error_list.append(error.detach().cpu())
        # print(len(y_list),y_list[0].shape,y_hat_list[0].shape)  #4 torch.Size([16, 5, 17])   | 53 torch.Size([1, 5, 17]) torch.Size([1, 5, 17])
        y_list = torch.cat(y_list, dim=0)
        y_hat_list = torch.cat(y_hat_list, dim=0)
        error_list = torch.cat(error_list, dim=0)
        
        return (y_list,y_hat_list,error_list)

    def plot_test(self,version="before",iterative_forecast=False):
        y, y_hat, error =  self.get_test_plot_data(iterative_forecast)
        if self.n_forecasts> 1:
            fig, axes = plt.subplots(self.out_features, 1, figsize=(30,4*self.out_features),constrained_layout=True)
            axes = axes.ravel()
            for (feat_nr,ax) in zip(range(self.out_features),axes):
                ax.plot(y[:,feat_nr],label="Target output",linewidth=1)
                ax.plot(y_hat[:,feat_nr], label="NCP output",linewidth=1)
                ax.set_title(f"{self.feature_labels[feat_nr]}",loc = 'left')
                ax.legend(loc='upper right')

            plt.suptitle(f"{version} training")
            plt.savefig(f"results/{self.task}/{self.model_id}_{version}_all_predictions.jpg")
            plt.close()


            fig, axes = plt.subplots(self.out_features, 1, figsize=(30,4*self.out_features),constrained_layout=True)
            axes = axes.ravel()
            for (feat_nr,ax) in zip(range(self.out_features),axes):
                ax.plot(error[:,feat_nr], label="Prediction error",linewidth=1)
                ax.set_title(f"{self.feature_labels[feat_nr]}",loc = 'left')
                ax.legend(loc='upper right')

            plt.suptitle(f"{version} training")
            plt.savefig(f"results/{self.task}/{self.model_id}_{version}_all_predictions-error.jpg")
            plt.close()

        pool = multiprocessing.Pool(processes=self.n_forecasts)
        inputs = list([(future_point,y,y_hat,error,self.n_forecasts,version,self.out_features,self.feature_labels,self.task,self.model_id) for future_point in range(self.n_forecasts)])
        pool.starmap(plot_future_point,inputs)
        return

def plot_future_point(future_point,y,y_hat,error,n_forecasts,version,out_features,feature_labels,task,model_id):
    future_point_indices = torch.LongTensor(list(range(future_point,y.shape[0],n_forecasts)))
    y_for_future = np.take(y, future_point_indices, 0)
    y_hat_for_future = np.take(y_hat,future_point_indices, 0)
    error_for_future = np.take(error, future_point_indices, 0)

    # plot the predictions for each feature in a subplot
    fig, axes = plt.subplots(out_features, 1, figsize=(30,4*out_features),constrained_layout=True)
    axes = axes.ravel()
    for (feat_nr,ax) in zip(range(out_features),axes):
        ax.plot(y_for_future[:,feat_nr],label="Target output",linewidth=1)
        ax.plot(y_hat_for_future[:,feat_nr], label="NCP output",linewidth=1)
        ax.set_title(f"{feature_labels[feat_nr]}",loc = 'left')
        ax.legend(loc='upper right')

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

    plt.suptitle(f"{version} training")
    plt.savefig(f"results/{task}/{model_id}_{version}_{future_point}-error.jpg")
    plt.close()
    print(f"saved plots in results/{task}/{model_id}_{version}")


data_classes = {
    "cheetah": CheetahData,
    "traffic": TrafficData,
    "occupancy": OccupancyData,
    "neurons": NeuronData,
    "neuronlaser": NeuronLaserData
}

study_names = {
    "cheetah":2024070307,
    "occupancy":20240626131851
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="ltc")
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--mixed_memory',action='store_true')
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--gpus', nargs='+', type=int,default = None)    
    parser.add_argument('--initial_lr',default=0.02,type=float)
    parser.add_argument('--cosine_lr',action='store_true')
    parser.add_argument('--dataset',default="cheetah",type=str) 
    parser.add_argument('--seq_len',default=32,type=int)
    parser.add_argument('--future',default=1,type=int)
    parser.add_argument('--iterative_forecast',action='store_true')
    parser.add_argument('--optimise',action='store_true')
    parser.add_argument('--pruning',action='store_true')
    parser.add_argument('--model_id_shift',type=int,default=0)
    parser.add_argument('--future_loss',action='store_true')
    parser.add_argument('--binwidth',default =0.05, type= float)
    parser.add_argument('--model_id',default =0, type= int)
    parser.add_argument('--checkpoint_id',default =0, type= int)
    parser.add_argument('--reset',action='store_true')

    args = parser.parse_args()


    assert args.future > 0 , "Future should be > 0"
    some_data_class = data_classes[args.dataset]
    dataset_data = some_data_class(future=args.future,seq_len=args.seq_len,binwidth=args.binwidth,iterative_forecast=args.iterative_forecast)
    task = args.dataset + "_forecast"
    checkpoint_id = None
    if not args.model_id:
        model_id = str(int(dt.datetime.today().strftime("%Y%m%d%H"))  + args.model_id_shift)
    else:
        model_id = args.model_id
    if args.checkpoint_id :# we copy checkpoint to new model
        checkpoint_id = args.checkpoint_id

    
    print(f" --------- model id: {model_id} --------- ")
    
    model = ForecastModel(task=task,model_id = model_id,model_type=args.model,_data=dataset_data,
        mixed_memory=args.mixed_memory,model_size=args.size,checkpoint_id=checkpoint_id)
    if not args.optimise:
        model.fit(epochs=args.epochs,gpus=args.gpus,optimise=args.optimise, learning_rate=args.initial_lr,cosine_lr=args.cosine_lr,future_loss=args.future_loss,reset = args.reset)
        model.test(iterative_forecast=args.iterative_forecast,checkpoint="last")
        model.test(iterative_forecast=args.iterative_forecast,checkpoint="best")
    
    else:
        storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{task}.db",
                engine_kwargs={"connect_args": {"timeout": 100}},
        )
        
        pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(),patience=2) if args.pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner,
                                    study_name= str(int(model_id)),
                                    storage=storage,load_if_exists=True)
        study.optimize(lambda trial: model.fit(trial=trial, _data=dataset_data,epochs=args.epochs,gpus=args.gpus,optimise=args.optimise,
                                            learning_rate=args.initial_lr,cosine_lr=args.cosine_lr,future_loss=args.future_loss),
                        n_trials=100)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


