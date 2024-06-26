import numpy as np
import pandas as pd
import os
import os
import sys
import argparse
import datetime as dt
from ncps.torch import LTC
import pytorch_lightning as pl
from ncps.wirings import AutoNCP 

import torch
import torch.utils.data as data
from LTC_learner import SequenceLearner
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor



def cut_in_sequences(x,seq_len,inc=1,prognosis=1):
    sequences_x = []
    sequences_y = []
    # x: time series
    # y: time series shifted into future

    # every x array has an overlap of (seq_len - inc)
    for s in range(0,x.shape[0] - seq_len-prognosis,inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(x[start+prognosis:end+prognosis])
    return sequences_x,sequences_y

class CheetahData:
    def __init__(self,seq_len=32,future=1,batch_size=16):
        all_files = sorted([os.path.join("data/cheetah",d) for d in os.listdir("data/cheetah") if d.endswith(".npy")])

        train_files = all_files[15:25]
        test_files = all_files[5:15]
        valid_files = all_files[:5]

        self.seq_len = seq_len
        self.obs_size = 17
        self.batch_size=batch_size
        self.future = future

        self.train_x, self.train_y = self._load_files(train_files)
        self.test_x, self.test_y = self._load_files(test_files)
        self.valid_x, self.valid_y = self._load_files(valid_files)

        print("train_x.shape:",str(self.train_x.shape))
        print("train_y.shape:",str(self.train_y.shape))
        print("valid_x.shape:",str(self.valid_x.shape))
        print("valid_y.shape:",str(self.valid_y.shape))
        print("test_x.shape:",str(self.test_x.shape))
        print("test_y.shape:",str(self.test_y.shape))
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
        print("data for dataloader: ", x_data.shape,y_data.shape)
        return data.DataLoader(
            data.TensorDataset(x_data, y_data),
            batch_size = self.batch_size,
            shuffle=True if subset == "train" else False,
            num_workers = 4 if subset != "test" else 1,
        )
    
    def _load_files(self,files):
        all_x = []
        all_y = []
        for f in files:
           
            arr = np.load(f)
            arr = arr.astype(np.float32)
            x,y = cut_in_sequences(arr,self.seq_len,10,self.future)

            all_x.extend(x)
            all_y.extend(y)

        return np.stack(all_x,axis=1),np.stack(all_y,axis=1)

    def iterate_train(self,batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,permutation[start:end]]
            batch_y = self.train_y[:,permutation[start:end]]
            #batch = ()
            yield (batch_x,batch_y)

class CheetahModel:
    def __init__(self,in_features,out_features,model_size,task,model_type="ltc"):
        self.model_type = model_type
        self.task = task
        self.model_size = model_size

        if(model_type.startswith("ltc")):
            wiring = AutoNCP(model_size,out_features)
            self._model = LTC(in_features,wiring,batch_first=True)

    def fit(self,_data,epochs,learning_rate,cosine_lr,n_gpus=None):
            loss = torch.nn.MSELoss()
            self.learn = SequenceLearner(self._model,loss,learning_rate=learning_rate,cosine_lr=cosine_lr,
                                         _loaderfunc=_data.get_dataloader,n_iterations=(_data.batch_size * epochs))
            lr_logger = LearningRateMonitor(logging_interval='step')

            tensorboard_logger = TensorBoardLogger(save_dir="log",
                    name=f"{self.task}_{self.model_type}")
            
            tensorboard_logger.log_hyperparams({
                "lr": learning_rate,
                "cosine_lr": cosine_lr
            })

            self.trainer = pl.Trainer(
                logger = tensorboard_logger,
                max_epochs= epochs,
                gradient_clip_val=1,
                accelerator= 'cpu' if n_gpus is None else 'gpu',
                devices=n_gpus,
                callbacks=[lr_logger]
            )
            self.trainer.fit(self.learn)

    def test(self):
        self.trainer.test(self.learn)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="ltc")
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--n_gpus',default=None,type=int)
    parser.add_argument('--initial_lr',default=0.02,type=float)
    parser.add_argument('--cosine_lr',action='store_true')
    parser.add_argument('--task',default="cheetah",type=str) # example: cheetah_forecast
    parser.add_argument('--future',default=1,type=int)
    args = parser.parse_args()


    cheetah_data = CheetahData(future=args.future)
    model = CheetahModel(in_features=cheetah_data.in_features, out_features = cheetah_data.out_features,task=args.task,model_size=args.size,model_type=args.model)
    # model = CheetahModel(model_type = args.model,model_size=args.size,sparsity_level=args.sparsity)
    model.fit(_data=cheetah_data,epochs=args.epochs,learning_rate=args.initial_lr,cosine_lr=args.cosine_lr,n_gpus=args.n_gpus)
    model.test()
    # model.fit(cheetah_,epochs=args.epochs,log_period=args.log)

