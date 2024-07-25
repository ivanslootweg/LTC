#%matplotlib qt
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import traceback
import time


from forecast import get_database_class
from forecast import ForecastModel
from forecast import NeuronLaserData
from forecast import MSELossfuture
from LTC_learner import SequenceLearner
from load_data import create_realtime_bins
from create_dummy_data import create_dummy_data

import fcntl 
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, EVENT_TYPE_MODIFIED, EVENT_TYPE_CREATED
import threading
import datetime as dt
from threading import Timer


class DataHandler(FileSystemEventHandler):
    def __init__(self,recording_id,load_func):
        self.recording_id = recording_id
        self.raw_data = f"recordings/{recording_id}/activations.mat"
        self.neuron_data = f"recordings/{recording_id}/activations.npy"
        self.load_realtime_data = load_func
        self.last_called = None
        self.cooldown = 0.5

    def on_created(self, event):
        if event.src_path.endswith(self.raw_data):
            current_time = time.time()
            if self.last_called is None or (current_time - self.last_called) > self.cooldown:
                print("created")
                self.last_called = current_time
                self.handle_raw_data()

    def on_modified(self, event):
        if event.src_path.endswith(self.raw_data):
            current_time = time.time()
            if self.last_called is None or (current_time - self.last_called) > self.cooldown:
                print("modified")
                self.last_called = current_time
                self.handle_raw_data()


    def handle_raw_data(self):
        try:
            print("creating bins")
            create_realtime_bins(neuroncount=20,fname=self.recording_id)
            self.load_realtime_data(self.neuron_data)

        except Exception as e:
            print(f"Error handling file raw dta: {e}")    
            # traceback.print_exc()

class PredictHandler(FileSystemEventHandler):
    def __init__(self, predict_func,load_func,recording_id ):
        # self.neuron_data = f"recordings/{recording_id}/activations.npy"
        self.recording_id = recording_id
        # self.plot_data_pos = f"recordings/{recording_id}/plot_data_pos.npy"
        # self.plot_data_neg = f"recordings/{recording_id}/plot_data_neg.npy"
        self.predict = predict_func
        self.lock = threading.Lock()
        self.load_realtime_data = load_func

    def on_created(self, event):
        pass
        # if event.src_path.endswith(self.neuron_data):
        #     self.handle_neuron_data()

    def on_modified(self, event):
        if event.src_path.endswith(self.neuron_data):
            self.handle_neuron_data()

    def handle_neuron_data(self):
        try:
            # self.load?realtime?data()
            self.predict()
            # plot_data_pos, plot_data_neg = self.predict()
            # print(plot_data_pos)
            # with self.lock:   
            #     np.save(self.plot_data_pos, plot_data_pos)
            #     np.save(self.plot_data_neg, plot_data_neg)
            
        except Exception as e:
            print(f"Error handling file A: {e}")
            traceback.print_exc()
        
class PlotHandler(FileSystemEventHandler):
    def __init__(self, recording_id):
        self.lock = threading.Lock()
        self.plot_data_pos = f"recordings/{recording_id}/plot_data_pos.npy"
        self.plot_data_neg = f"recordings/{recording_id}/plot_data_neg.npy"

    def on_created(self, event):
        pass
        # if event.src_path.endswith(self.plot_data_pos): 
        #     self.handle_plot_data()

    def on_modified(self, event):
        if event.src_path.endswith(self.plot_data_pos): 
            self.handle_plot_data()


    def handle_plot_data(self):
        try:
            with self.lock:
                plot_data_pos = np.load(self.plot_data_pos)
                plot_data_neg = np.load(self.plot_data_neg)
            self.predictor.plot_realtime(plot_data_pos,plot_data_neg)
            
        except Exception as e:
            print(f"Error handling file C: {e}")    
            traceback.print_exc()
    
class RealtimeNeuronLaserData(NeuronLaserData):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.data_condition = threading.Condition()
        # super().load_data()

    def load_realtime_data(self,_path):
        x = np.load(_path)
        x = x.astype(np.float32)
        print("loading data",x.shape)
        x = np.transpose(x)
        
        with self.data_condition:
            setattr(self,f"predict_x",x)   
            self.data_condition.notify_all()
        return

class RealtimeForecastModel(ForecastModel):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.set_model()
        try:    
            self.learn = SequenceLearner.load_from_checkpoint(f"{self.load_dir}/best.ckpt",model=self._model)
        except: 
            self.learn = SequenceLearner.load_from_checkpoint(f"{self.load_dir}/best.ckpt",model=self._model)
        self.learn.eval()
        self.total_read_timesteps = 0
        self.trainer = pl.Trainer()
        self.load_realtime = kwargs["_data"].load_realtime_data
        self.recorded_history = torch.empty((0,self.in_features-1),dtype=torch.float32)
        self.predict_pos_history = torch.full((self.seq_len,self.in_features-1),0)
        self.predict_neg_history = torch.full((self.seq_len,self.in_features-1),0)
        self.plot_condition = threading.Condition()
        self.data_condition = kwargs["_data"].data_condition

    def run(self,recording_id):  
        # instantiate dual observer
        self.recording_id = recording_id
        data_handler = DataHandler(recording_id,self.load_realtime)
        # predict_handler = PredictHandler(self.predict,self.load_realtime,recording_id)
        # plot_handler = PlotHandler(recording_id)

        plot_thread = threading.Thread(target=self.plot_realtime)
        plot_thread.start()

        predict_thread = threading.Thread(target=self.predict)
        predict_thread.start()


        observer = Observer()
        observer.schedule(data_handler, path=f"recordings/{recording_id}", recursive=False)   
        # observer.schedule(predict_handler, path=f"recordings/{recording_id}", recursive=False)  
        # observer.schedule(plot_handler, path=f"recordings/{recording_id}", recursive=False) 
        observer.start()

        try:
            while True:
                pass
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


    def plot_realtime(self,version="realtime"):
        while True:
            with self.plot_condition:
                self.plot_condition.wait()
                print("plotting realtime")
                plot_length = min(64,self.total_read_timesteps)
                plot_pos = self.predict_pos_history[-plot_length:] 
                plot_neg = self.predict_neg_history[-plot_length:]
                history = self.recorded_history[-plot_length:]
                fig, axes = plt.subplots(self.out_features, 1, figsize=(30,4*self.out_features),constrained_layout=True)
                axes = axes.ravel()
                for (feat_nr,ax) in zip(range(self.out_features),axes):
                    ax.plot(plot_pos[:,feat_nr],label="With activation",linewidth=1)
                    ax.plot(plot_neg[:,feat_nr], label="No activation",linewidth=1)
                    ax.plot(history[:,feat_nr], label="Recording",linewidth=1)
                    ax.set_title(f"{self.feature_labels[feat_nr]}",loc = 'left')
                    ax.legend(loc='upper right')

                plt.suptitle(f"{version} predictions")
                plt.savefig(f"recording/{self.recording_id}.jpg")
                # plt.show()
                plt.close()
                return

    def predict(self):
        device = next(self._model.parameters()).device
        while True:
            with self.data_condition:
                self.data_condition.wait()
                print("predicting realtime")
                in_x, in_x_de = self._loaderfunc(subset="predict")
                if in_x.shape[1] < self.seq_len:
                    print("too small",in_x.shape)
                    return
                in_x.permute(1,0,2) # shape of (seq_len,1,neurons) > (1,seq_len,neurons) 
                new_read_timesteps = in_x.shape[1] - self.total_read_timesteps
                self.total_read_timesteps += new_read_timesteps
                in_x = in_x[:,-self.seq_len:,:] #fetch last 32 timestamps
                """Here we add the recordings to the history"""
                self.recorded_history = torch.concat((self.recorded_history,torch.tensor(in_x_de[-new_read_timesteps:])), dim= 0) # 
                x_pos = x_neg = in_x # (1,seq_len,neurons)
                y_hat_pos = y_hat_neg  = torch.empty((1,self.n_iterative_forecasts,in_x.shape[-1]), device=in_x.device)
                for i in range(self.n_iterative_forecasts):
                    """we only add a 1 for laser activation at the 2 direct following bin, no  other bins"""
                    x_pos_in = torch.cat((x_pos,torch.full((1,32,1),1 if i < 2 else 0)),dim=-1)
                    x_neg_in = torch.cat((x_neg,torch.full((1,32,1), 0)),dim=-1)
                    x = torch.cat((x_pos_in,x_neg_in),dim=0).to(device)
                    pred, _ = self._model.forward(x) # (2, last timestep, neurons(excluding activation))
                    (next_step_pos,next_step_neg) = pred.detach().cpu()
                    y_hat_pos[:,i] = next_step_pos[-1:,:]
                    y_hat_neg[:,i] = next_step_neg[-1:,:]
                    x_pos = torch.cat((x_pos[:,1:],next_step_pos[-1:,:].unsqueeze(0)),dim=1)
                    x_neg = torch.cat((x_neg[:,1:],next_step_neg[-1:,:].unsqueeze(0)),dim=1) 

                """TODO Now we compare the pos (stimulation) predictions and neg (absence) predictions"""
                """Now we plot the recorded history of the signal followed by the forecasts
                        for every neuron + activation
                        make sure to include the laseractivation feature in the predictions
                """
                y_hat_pos_de = self.denormalize(y_hat_pos,"y").flatten(0,1) # (5,17)
                y_hat_neg_de = self.denormalize(y_hat_neg,"y").flatten(0,1)

            with self.plot_condition:
                print("setting plot data")
                self.predict_pos_history = torch.concat((self.predict_pos_history,y_hat_pos_de), dim= 0) 
                self.predict_neg_history = torch.concat((self.predict_neg_history,y_hat_neg_de), dim= 0) 
                self.plot_condition.notify_all()
                # return(y_hat_pos_de,y_hat_neg_de)
                return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id',default =0, type= int)
    parser.add_argument('--model',default="ltc")
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=0,type=int)
    parser.add_argument('--gpus', nargs='+', type=int,default = None)    
    parser.add_argument('--seq_len',default=32,type=int)
    parser.add_argument('--future',default=1,type=int)

    args = parser.parse_args()
    iterative_forecast = True
    model = "ltc"
    epochs = 0
    mixed_memory = True
    task =  "neuronlaser_forecast"
    recording_id = str(int(dt.datetime.today().strftime("%Y%m%d%H%M")))
       
    assert args.future > 0 , "Future should be > 0"
    some_data_class = get_database_class(RealtimeNeuronLaserData)
    dataset_data = some_data_class(future=args.future,seq_len=args.seq_len,iterative_forecast=iterative_forecast)
    model_id = args.model_id
    recording_id = "dummy"
    print(f" --------- model id: {model_id} recording id : {recording_id}--------- ")
    model = RealtimeForecastModel(model_id = model_id,model_type=model,mixed_memory=mixed_memory,_data=dataset_data,model_size=args.size,task=task)
    
    dummy_data_thread = threading.Thread(target=create_dummy_data, args=(recording_id,))
    dummy_data_thread.start()
    # Run the main model's run method
    model.run(recording_id)

    # Wait for the dummy data thread to finish if needed
    dummy_data_thread.join()
