import numpy as np
import pandas as pd
import os
import os
import sys
import argparse
import datetime as dt
from ncps.torch import LTC
import pytorch_lightning as pl
from ncps.wirings import AutoNCP, FullyConnected

import torch
import torch.nn as nn
import torch.utils.data as data
from LTC_learner import SequenceLearner
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint


import optuna
from optuna.integration import PyTorchLightningPruningCallback

import matplotlib.pyplot as plt


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, inputs, targets):
        # normalzed mean squared error
        loss = torch.mean(((targets-inputs)**2))/ torch.mean(targets)
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

class TrafficData:
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


class OccupancyData:
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
            

class CheetahData:
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


class NeuronData:
    def __init__(self,seq_len=32,future=1,batch_size=16):
        x, y = load_trace()

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.future = future
        """ 
        x: (20,T) -> transpose to (T,20)
        train_x after cut : (?) * (32,7)
        after stack: (T, 32, 7). should be (32,T,7)
        """
        x = np.load("data/neurons/activations.npy")
        x = x.astype(np.float32)
        x = np.transpose(x)
        train_x, train_y = cut_in_sequences(x,seq_len=seq_len, inc=2,prognosis=future)
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

        self.feature_labels = self.feature_labels = [f"neuron {i}" for i in range(self.train_x.shape[2])]


def get_database_class(data_base):
    class DataBaseClass(data_base):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            dataset_names = ["train_x","train_y","valid_x", "valid_y","test_x","test_y"]
            train_with_noise = False
            for _name in dataset_names:
                _array = getattr(self,_name)
                print(f"{_name} shape: ", str(_array.shape), str(_array.mean()))
                _array = self.normalize(_array,_name.split("_")[-1])
                print(f"{_name} post normalisation: ",_array.mean())
                if "train" in _name and train_with_noise:
                    noise = np.random.normal(0, 0.1, _array.shape) 
                    _array = _array + noise
                    setattr(self,_name, _array)
                    print(f"{_name} post noise: ",str(_array.mean()))              

            self.in_features = self.train_x.shape[2]
            self.out_features = self.train_y.shape[2]

        def get_dataloader(self,subset="train"):
            # dataloader input of shapes [BATCH,series_length,features]
            # _x.shape is currently: (32, NSERIES, 17)
            # _y.shape is currently: (32, NSERIES, 17)
            # I do not want to change the existing class methods if not needed
            # we permute the data to (NSERIES,32,17)
            assert (subset) in ["train","valid","test"]
            x_data = torch.permute(torch.tensor(getattr(self,f"{subset}_x")),(1,0,2))
            y_data = torch.permute(torch.tensor(getattr(self,f"{subset}_y")),(1,0,2))
            return data.DataLoader(
                data.TensorDataset(x_data, y_data),
                batch_size = self.batch_size,
                shuffle=True if subset == "train" else False,
                num_workers = 4 if subset != "test" else 1,
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
            return (tensor - self.mean[values]) / self.std[values]
        
        def denormalize(self,tensor,values="x"):
            return tensor * self.std[values] + self.mean[values]
        
    return DataBaseClass


class ForecastModel:
    def __init__(self,task,model_id,model_type="ltc"):
        self.model_type = model_type
        self.task = task
        self.model_id = model_id

    def fit(self,trial=None,_data=None,epochs=100,learning_rate=1e-2,cosine_lr=False,model_size=None,optimise=False,gpus=None,mixed_memory=False):
            loss = torch.nn.MSELoss()
            # loss = NMSELoss()
            self.n_epochs = epochs
            self.future = _data.future
            self.feature_labels = _data.feature_labels
            self.in_features = _data.in_features
            self.out_features = _data.out_features
            self.mean = _data.mean
            self.std = _data.std
            self.experiment_name = f"{self.task}_{self.model_id}"

            if not optimise:
                self.model_size = model_size
                if(self.model_type.startswith("ltc")):
                    # wiring = AutoNCP(model_size,out_features)
                    wiring = FullyConnected(model_size,self.out_features)
                    self._model = LTC(self.in_features,wiring,batch_first=True,mixed_memory=mixed_memory)
                self.learn = SequenceLearner(self._model,loss,learning_rate=learning_rate,cosine_lr=cosine_lr,
                                _loaderfunc=_data.get_dataloader,n_iterations=(_data.batch_size * epochs))
                lr_logger = LearningRateMonitor(logging_interval='step')
                            
                tensorboard_logger = TensorBoardLogger(save_dir="log",
                        version = f"{self.model_id}",
                        name=f"{self.task}_{self.model_type}")
                
                tensorboard_logger.log_hyperparams({
                    "lr": learning_rate,
                    "cosine_lr": cosine_lr,
                    "seq_len":_data.seq_len,
                    "future": _data.future,
                    "model_size": model_size
                })

                self.trainer = pl.Trainer(
                    logger = tensorboard_logger,
                    max_epochs= epochs,
                    gradient_clip_val=1,
                    accelerator= 'cpu' if gpus is None else 'gpu',
                    devices=gpus,
                    callbacks=[lr_logger],
                )

            else:
                learning_rate = trial.suggest_float("learning_rate",1e-4,1e-2)
                cosine_lr = trial.suggest_int("cosine_lr",0,1) if cosine_lr else 0
                model_size = trial.suggest_int("model_size",self.out_features +3,48,4)
                checkpoint_callback = ModelCheckpoint(
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    dirpath=f"./checkpoints/{self.task}_{self.model_id}",
                    filename="{epoch:02d}-{val_loss:.2f}",
                )

                self.model_size = model_size
                if(self.model_type.startswith("ltc")):
                    print("model size: ", model_size, "out: ", self.out_features)
                    # wiring = AutoNCP(model_size,self.out_features)
                    wiring = FullyConnected(model_size,self.out_features)
                    self._model = LTC(self.in_features,wiring,batch_first=True,mixed_memory=mixed_memory)
                self.learn = SequenceLearner(self._model,loss,learning_rate=learning_rate,cosine_lr=cosine_lr,
                                _loaderfunc=_data.get_dataloader,n_iterations=(_data.batch_size * epochs))
                self.trainer = pl.Trainer(
                    logger = True,
                    max_epochs= epochs,
                    gradient_clip_val=1,
                    accelerator= 'cpu' if gpus is None else 'gpu',
                    devices=gpus,
                    callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss"),checkpoint_callback],

                )

                hyperparameters = dict(lr=learning_rate, seq_len=_data.seq_len, future=_data.future,
                                       model_size = model_size)
                self.trainer.logger.log_hyperparams(hyperparameters)

            self.trainer.fit(self.learn)
        
            if optimise:
               return self.trainer.callback_metrics["val_loss"].item()
            
    def denormalize(self,tensor,values="x"):
        return tensor * self.std[values] + self.mean[values]

    def test(self):
        self.trainer.test(self.learn)
        if self.n_epochs > 1: 
            self.plot_test(version="after")
        else:
            self.plot_test(version="before")

    def plot_test(self,version="before"):
        y_list = []
        y_hat_list = []
        error_list = []

        for _batch in self.trainer.test_dataloaders:
            # _batch = (B_size, Timesteps, Features)
            x, y = _batch
            y_hat, _ = self._model.forward(x)
            y_hat = y_hat.view_as(y)
            y = self.denormalize(y,"y")
            y_hat = self.denormalize(y_hat,"y")
            error = y - y_hat
            y_list.extend(y[:,-self.future:,:])
            y_hat_list.extend(y_hat[:,-self.future:,:])
            error_list.extend(error[:,-self.future:,:])

        y = torch.cat(y_list,dim=0).detach().numpy()
        y_hat = torch.cat(y_hat_list,dim=0).detach().numpy()
        error = torch.cat(error_list,dim=0).detach().numpy()
        print(f" y {y.shape} error {error.shape}")
        fig, axes = plt.subplots(self.in_features, 1, figsize=(30,4*self.in_features),constrained_layout=True)
        axes = axes.ravel()
        for (feat_nr,ax) in zip(range(self.in_features),axes):
            ax.plot(y[:,feat_nr],label="Target output",linewidth=1)
            ax.plot(y_hat[:,feat_nr], label="NCP output",linewidth=1)
            ax.set_title(f"{self.feature_labels[feat_nr]}",loc = 'left')
            ax.legend(loc='upper right')

        plt.suptitle(f"{version} training")
        plt.savefig(f"results/{self.task}_{self.model_id}_{version}_all_predictions.jpg")
        plt.close()

        fig, axes = plt.subplots(self.in_features, 1, figsize=(30,4*self.in_features),constrained_layout=True)
        axes = axes.ravel()
        for (feat_nr,ax) in zip(range(self.in_features),axes):
            ax.plot(error[:,feat_nr], label="Prediction error",linewidth=1)
            ax.set_title(f"{self.feature_labels[feat_nr]}",loc = 'left')
            ax.legend(loc='upper right')

        plt.suptitle(f"{version} training")
        plt.savefig(f"results/{self.task}_{self.model_id}_{version}_all=predictionss-error.jpg")
        plt.close()
        
        for future_point in range(self.future):
            future_point_indices = torch.LongTensor(list(range(0+future_point,y.shape[0]-self.future +future_point +1,self.future)))
            y_for_future = np.take(y, future_point_indices, 0)
            y_hat_for_future = np.take(y_hat,future_point_indices, 0)
            error_for_future = np.take(error, future_point_indices, 0)
            # plot the predictions for each feature in a subplot
            print(f" y_for_future {y_for_future.shape} error_for_future {error_for_future.shape}")
            fig, axes = plt.subplots(self.in_features, 1, figsize=(30,4*self.in_features),constrained_layout=True)
            axes = axes.ravel()
            for (feat_nr,ax) in zip(range(self.in_features),axes):
                ax.plot(y_for_future[:,feat_nr],label="Target output",linewidth=1)
                ax.plot(y_hat_for_future[:,feat_nr], label="NCP output",linewidth=1)
                ax.set_title(f"{self.feature_labels[feat_nr]}",loc = 'left')
                ax.legend(loc='upper right')

            plt.suptitle(f"{version} training")
            plt.savefig(f"results/{self.task}_{self.model_id}_{version}_{future_point}.jpg")
            plt.close()

            # plot the error for each feature in a subplot
            fig, axes = plt.subplots(self.in_features, 1, figsize=(30,4*self.in_features),constrained_layout=True)
            axes = axes.ravel()
            for (feat_nr,ax) in zip(range(self.in_features),axes):
                ax.plot(error_for_future[:,feat_nr], label="Prediction error",linewidth=1)
                ax.set_title(f"{self.feature_labels[feat_nr]}",loc = 'left')
                ax.legend(loc='upper right')

            plt.suptitle(f"{version} training")
            plt.savefig(f"results/{self.task}_{self.model_id}_{version}_{future_point}-error.jpg")
            plt.close()

data_classes = {
    "cheetah": CheetahData,
    "traffic": TrafficData,
    "occupancy": OccupancyData,
    "neurons": NeuronData
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
    parser.add_argument('--optimise',action='store_true')
    parser.add_argument('--pruning',action='store_true')
    parser.add_argument('--load_previous',type=int,default=0)

    args = parser.parse_args()

    some_data_class = get_database_class(data_classes[args.dataset])
    dataset_data = some_data_class(future=args.future,seq_len=args.seq_len)

    # if args.future > 1:
    task = args.dataset + "_forecast"
    model_id = dt.datetime.today().strftime("%Y%m%d%H")

    print(f" --------- model id: {model_id} --------- ")
    
    
    model = ForecastModel(task=task,model_id = model_id,model_type=args.model)
    if not args.optimise:
        model.fit(_data=dataset_data,epochs=args.epochs,gpus=args.gpus,optimise=args.optimise,mixed_memory=args.mixed_memory,
                learning_rate=args.initial_lr,cosine_lr=args.cosine_lr,model_size=args.size)
        model.test()
    
    else:
        storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{task}.db",
                engine_kwargs={"connect_args": {"timeout": 100}},
        )
        
        pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner,
                                    study_name= str(int(model_id)-args.load_previous),
                                    storage=storage,load_if_exists=True)
        study.optimize(lambda trial: model.fit(trial=trial,  
                                            _data=dataset_data,epochs=args.epochs,gpus=args.gpus,optimise=args.optimise,mixed_memory=args.mixed_memory,
                                            learning_rate=args.initial_lr,cosine_lr=args.cosine_lr,model_size=args.size),
                        n_trials=100)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


