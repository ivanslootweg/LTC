#%matplotlib qt
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import traceback
import time
import contextlib
import multiprocessing
from multiprocessing import Pipe, Manager
from multiprocessing.managers import BaseManager, NamespaceProxy

from forecast import ForecastModel
from forecast import NeuronLaserData,  DataBaseClass
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
from matplotlib.animation import FuncAnimation


plt.rcParams.update({
    'axes.titlesize': 4,    # Title size
    'axes.labelsize': 4,     # Axis label size
    'xtick.labelsize':4,    # X-axis tick label size
    'ytick.labelsize':4,    # Y-axis tick label size
    'legend.fontsize':4,    # Legend font size
    'figure.titlesize' :4
})

multiprocessing.set_start_method('spawn', force=True)


class PauseDataHandler(FileSystemEventHandler):
    def __init__(self,recording_id,load_func):
        self.recording_id = recording_id
        self.raw_data = f"recordings/{recording_id}/activations.mat"
        self.neuron_data = f"recordings/{recording_id}/activations.npy"
        self.load_realtime_data = load_func
        self.last_called = None
        self.cooldown = 0.5
        self.include_from = 0

    def on_created(self, event):
        print("created",flush=True  )

        pass

    def on_modified(self, event):
        if event.src_path.endswith(self.raw_data):
            with self.ignore_events() as can_enter:
                if can_enter:
                    print("modified",flush=True)
                    self.handle_raw_data()
        return

    def handle_raw_data(self):
        try:
            print("creating bins",flush=True)
            self.include_from = create_realtime_bins(neuroncount=20,fname=self.recording_id,include_from =self.include_from)
            self.load_realtime_data(self.neuron_data)

        except Exception as e:
            print(f"Error handling file raw data: {e}")    
            traceback.print_exc()

    @contextlib.contextmanager
    def ignore_events(self):
        current_time = time.time()
        if self.last_called is None or (current_time - self.last_called) > self.cooldown:
            self.last_called = current_time
            yield True
        else:
            yield False


class DataHandler(FileSystemEventHandler):
    def __init__(self,recording_id,load_func):
        self.recording_id = recording_id
        self.raw_data = f"recordings/{recording_id}/activations.mat"
        self.neuron_data = f"recordings/{recording_id}/activations.npy"
        self.load_realtime_data = load_func
        self.last_called = None
        self.cooldown = 0.5
        self.include_from = 0

    def on_created(self, event):
        if event.src_path.endswith(self.raw_data):
            current_time = time.time()
            if self.last_called is None or (current_time - self.last_called) > self.cooldown:
                print("created",flush=True)
                self.last_called = current_time
                self.handle_raw_data()

    def on_modified(self, event):
        if event.src_path.endswith(self.raw_data):
            current_time = time.time()
            if self.last_called is None or (current_time - self.last_called) > self.cooldown:
                print("modified",flush=True)
                self.last_called = current_time
                self.handle_raw_data()
            return


    def handle_raw_data(self):
        try:
            print("creating bins",flush=True)
            self.include_from = create_realtime_bins(neuroncount=20,fname=self.recording_id,include_from =self.include_from)
            self.load_realtime_data(self.neuron_data)
            return
        except Exception as e:
            print(f"Error handling file raw data: {e}")    
            traceback.print_exc()

class PredictHandler(FileSystemEventHandler):
    def __init__(self, predict_func,load_func,recording_id ):
        self.recording_id = recording_id
        self.predict = predict_func
        self.lock = threading.Lock()
        self.load_realtime_data = load_func

    def on_created(self, event):
        pass

    def on_modified(self, event):
        if event.src_path.endswith(self.neuron_data):
            self.handle_neuron_data()

    def handle_neuron_data(self):
        try:
            self.predict()

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
    def __init__(self,pipe_connection,**kwargs):
        super().__init__(**kwargs)
        # self.data_condition = threading.Condition()
        self.pipe_connection = pipe_connection

    def create(cls, *args, **kwargs):
        class_str = "RealtimeNeuronLaserData"
        BaseManager.register(class_str, cls, ObjProxy, exposed=tuple(dir(cls)))

        # Start a manager process
        manager = BaseManager()
        manager.start()

        # Create and return this proxy instance. Using this proxy allows sharing of state between processes.
        inst = eval("manager.{}(*args, **kwargs)".format(class_str))
        return inst

    def load_realtime_data(self,_path):
        x = np.load(_path)
        x = x.astype(np.float32)
        x = np.transpose(x)
        
        # with self.data_condition:
        #     setattr(self,f"predict_x",x)   
        #     print("notified predict",flush=True)
        #     self.data_condition.notify_all()
        #     return

        setattr(self,f"predict_x",x)   
        t = time.time()
        print(f"send {t}",flush=True)
        self.pipe_connection.send(t)
    
    def set_pipe(pipe_connection):
        self.pipe_connection = pipe_connection

class RealtimeForecastModel(ForecastModel):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.set_model()
        self.total_read_timesteps = 0
        self.trainer = pl.Trainer()
        self.load_realtime = kwargs["_data"].load_realtime_data
        self.norm_recorded_history = torch.empty((1,0,self.in_features-1),dtype=torch.float32)
        self.recorded_history = torch.empty((0,self.in_features-1),dtype=torch.float32)
        self.predict_pos_history = torch.full((self.n_forecasts,self.in_features-1),0)
        self.predict_neg_history = torch.full((self.n_forecasts,self.in_features-1),0)
        # self.data_condition = kwargs["_data"].data_condition

        # self.fig, self.axes = plt.subplots(int(self.out_features/2), 2, figsize=(30,10*self.out_features),constrained_layout=True)
        # self.axes = self.axes.ravel()

        # self.lines = []
        # for feat_nr, ax in enumerate(self.axes):
        #     ax.set_ylim(-0.1,None) 
        #     ax.set_xlim(0,64)
        #     line_pos, = ax.plot([], [], lw=0.5, label="With activation")
        #     line_neg, = ax.plot([], [], lw=0.5, label="No activation")
        #     line_rec, = ax.plot([], [], lw=0.5, label="Recording")
        #     self.lines.append((line_pos, line_neg, line_rec))
        #     ax.legend(loc='upper right')

        # self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000) #run in main thread


    def run(self,recording_id,gpus,pipe_connection):  
        try:    
            self.learn = SequenceLearner.load_from_checkpoint(f"{self.load_dir}/best.ckpt",model=self._model,map_location=torch.device(gpus))
        except: 
            self.learn = SequenceLearner.load_from_checkpoint(f"{self.load_dir}/best.ckpt",model=self._model,map_location=torch.device(gpus))
        self.learn.eval()
        self.pipe_connection = pipe_connection
        self.recording_id = recording_id

        self.predict()


    def update_plot(self, *args):
        plot_length = min(64, self.total_read_timesteps)
        if plot_length == 0:
            return self.lines  # No data to plot yet

        plot_pos = self.predict_pos_history[-(plot_length+self.n_forecasts):].numpy()
        plot_neg = self.predict_neg_history[-(plot_length+self.n_forecasts):].numpy()
        history = self.recorded_history[-plot_length:].numpy()
        print("plotting now: ", history.shape, plot_pos.shape,plot_neg.shape,flush=True)
        for i, (line_pos, line_neg, line_rec) in enumerate(self.lines):
            line_pos.set_data(range(plot_pos.shape[0]), plot_pos[:, i])
            line_neg.set_data(range(plot_neg.shape[0]), plot_neg[:, i])
            line_rec.set_data(range(history.shape[0]), history[:, i])
            self.axes[i].relim()
        # self.axes[i].autoscale_view()

        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


    def predict(self):
        self.device = next(self._model.parameters()).device
        # while not self.stop_event.is_set():
        self.norm_recorded_history = self.norm_recorded_history.to(self.device,non_blocking=True)
        print("start predict")
        while True:
            # with self.data_condition:
            #     self.data_condition.wait()
            if  not self.pipe_connection.poll():
                continue
            _ = self.pipe_connection.recv()
            print(f"read {_}")
            in_x, in_x_de = self._loaderfunc(subset="predict")
            # print("incoming: ",in_x.shape,flush=True)
            in_x = in_x.to(self.device,non_blocking=True) # shape of (seq_len,1,neurons) > (1,seq_len,neurons) 
            print("incoming: ",in_x.shape,flush=True)
            # new_read_timesteps = in_x.shape[1] - self.total_read_timesteps
            new_read_timesteps = in_x.shape[1]
            missed_predictions = max(0,new_read_timesteps - self.n_forecasts)
            self.total_read_timesteps += new_read_timesteps
            self.norm_recorded_history = torch.concat((self.norm_recorded_history,in_x), dim= 1) 
            
            if self.total_read_timesteps < self.seq_len:
                print("too small",in_x.shape,flush=True)
                continue
            # in_x = in_x[:,-self.seq_len:,:] #fetch last 32 timestamps
            in_x = self.norm_recorded_history[:,-self.seq_len:,:]
            x_pos = x_neg = in_x # (1,seq_len,neurons)
            y_hat_pos = y_hat_neg  = torch.empty((1,self.n_iterative_forecasts,in_x.shape[-1]), device=self.device)
            # _start = dt.datetime.now()
            for i in range(self.n_iterative_forecasts):
                """we only add a 1 for laser activation at the 2 direct following bin, no  other bins"""
                x_pos_in = torch.cat((x_pos,torch.full((1,32,1),1 if i < 2 else 0,device=self.device)),dim=-1)
                x_neg_in = torch.cat((x_neg,torch.full((1,32,1), 0,device=self.device)),dim=-1)
                x = torch.cat((x_pos_in,x_neg_in),dim=0)
                # print("before model call")
                # print(f"pred {i}",flush=True)
                pred, _ = self._model.forward(x) # (2, last timestep, neurons(excluding activation))
                (next_step_pos,next_step_neg) = pred
                y_hat_pos[:,i] = next_step_pos[-1:,:]
                y_hat_neg[:,i] = next_step_neg[-1:,:]
                x_pos = torch.cat((x_pos[:,1:],next_step_pos[-1:,:].unsqueeze(0)),dim=1)
                x_neg = torch.cat((x_neg[:,1:],next_step_neg[-1:,:].unsqueeze(0)),dim=1) 
            """TODO Now we compare the pos (stimulation) predictioWns and neg (absence) predictions"""
            """Now we plot the recorded history of the signal followed by the forecasts
                    for every neuron + activation
                    make sure to include the laseractivation feature in the predictions
            """
            # print('Duration: {}'.format(dt.datetime.now() - _start),flush=True)
            y_hat_pos_de = self.denormalize(y_hat_pos.detach().cpu(),"y").flatten(0,1) # (5,17)
            y_hat_neg_de = self.denormalize(y_hat_neg.detach().cpu(),"y").flatten(0,1)
        
            # self.y_hat_pos_de = y_hat_pos_de
            # self.y_hat_neg_de = y_hat_neg_de
            self.recorded_history = torch.concat((self.recorded_history,torch.tensor(in_x_de[-new_read_timesteps:])), dim= 0) 
            self.predict_pos_history = torch.concat((self.predict_pos_history,torch.full((missed_predictions,self.in_features-1),0),y_hat_pos_de), dim= 0) 
            self.predict_neg_history = torch.concat((self.predict_neg_history,torch.full((missed_predictions,self.in_features-1),0),y_hat_neg_de), dim= 0) 
            print("records so far: ", self.recorded_history.shape, self.predict_pos_history.shape,self.predict_neg_history.shape,flush=True)
            continue

def run(model_id,model_type,mixed_memory,task,recording_id,parent_connection,iterative_forecast,child_connection,inst,args):
    # dataset_data = RealtimeNeuronLaserData(future=args.future,seq_len=args.seq_len,iterative_forecast=iterative_forecast)
    # dataset_data.pipe_connection = child_connection
    print(inst.pipe_connection)
    model = RealtimeForecastModel(model_id = model_id,model_type=model_type,mixed_memory=mixed_memory,_data=inst,model_size=args.size,task=task)
    model.run(recording_id,gpus=args.gpus[0],pipe_connection = parent_connection)


class ObjProxy(NamespaceProxy):
    """Returns a proxy instance for any user defined data-type. The proxy instance will have the namespace and
    functions of the data-type (except private/protected callables/attributes). Furthermore, the proxy will be
    pickable and can its state can be shared among different processes. """

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                return self._callmethod(name, args, kwargs)
            return wrapper
        return result


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
    model_type = "ltc"
    epochs = 0
    mixed_memory = True
    task =  "neuronlaser_forecast"
    recording_id = str(int(dt.datetime.today().strftime("%Y%m%d%H%M")))
       
    assert args.future > 0 , "Future should be > 0"
    # some_data_class = get_database_class(RealtimeNeuronLaserData)
    # dataset_data = RealtimeNeuronLaserData(future=args.future,seq_len=args.seq_len,iterative_forecast=iterative_forecast)

    
    model_id = args.model_id
    recording_id = "dummy"
    print(f" --------- model id: {model_id} recording id : {recording_id}--------- ")
    # Run the main model's run method

    parent_connection, child_connection = Pipe()
    # dataset_data.pipe_connection = child_connection


    # BaseManager.register('RealtimeNeuronLaserData', RealtimeNeuronLaserData)
    # manager = BaseManager()
    # manager.start()
    inst= RealtimeNeuronLaserData.create(child_connection,future=args.future,seq_len=args.seq_len,iterative_forecast=iterative_forecast)

    processes = []
    data_observer = Observer()

    data_observer.daemon = True 
    data_handler = PauseDataHandler(recording_id,inst.load_realtime_data)
    data_observer.schedule(data_handler, path=f"recordings/{recording_id}", recursive=False)   
    data_observer.start()

    # dummy_data_thread = threading.Thread(target=create_dummy_data, args=(recording_id,))
    # dummy_data_thread.daemon = True
    # dummy_data_thread.start()

    # predict_thread = threading.Thread(target=self.predict)
    # predict_thread.daemon = True
    # predict_thread.start()

    dummy_data_process = multiprocessing.Process(target=create_dummy_data, args=(recording_id,))    
    dummy_data_process.start()
    processes.append(dummy_data_process)

    predict_process = multiprocessing.Process(target=run,args=(model_id,model_type,mixed_memory,task,recording_id,parent_connection,iterative_forecast,child_connection,inst,args))    
    predict_process.start()
    processes.append(predict_process)


    try:
        # plt.show()
        while True:
            pass
    except KeyboardInterrupt:
        # plt.close('all')
        data_observer.stop()
        data_observer.join()
        for process in processes:
            process.terminate()

    # plt.close('all')
    data_observer.stop()
    data_observer.join()
    for process in processes:
        process.terminate()

