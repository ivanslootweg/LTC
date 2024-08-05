#%matplotlib qt
import argparse
import matplotlib.pyplot as plt
import numpy as np
import traceback
import time
import contextlib
import multiprocessing
from multiprocessing import Pipe


from load_data import create_realtime_bins
from create_dummy_data import create_dummy_data
from modules import RealtimeNeuronLaserData, RealtimeForecastModel

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import datetime as dt

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
    def __init__(self,recording_id,prepare_realtime_data):
        self.recording_id = recording_id
        self.raw_data = f"recordings/{recording_id}/activations.mat"
        self.neuron_data = f"recordings/{recording_id}/activations.npy"
        self.prepare_realtime_data = prepare_realtime_data
        self.last_called = None
        self.cooldown = 0.5
        self.include_from = 0

    def on_created(self, event):
        # print("created",flush=True  )

        pass

    def on_modified(self, event):
        if event.src_path.endswith(self.raw_data):
            with self.ignore_events() as can_enter:
                if can_enter:
                    # print("modified",flush=True)
                    self.handle_raw_data()
        return

    def handle_raw_data(self):
        try:
            # print("creating bins",flush=True)
            self.include_from = create_realtime_bins(neuroncount=20,fname=self.recording_id,include_from =self.include_from)
            self.prepare_realtime_data(self.neuron_data)

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
    def __init__(self,recording_id,prepare_realtime_data):
        self.recording_id = recording_id
        self.raw_data = f"recordings/{recording_id}/activations.mat"
        self.neuron_data = f"recordings/{recording_id}/activations.npy"
        self.prepare_realtime_data = prepare_realtime_data
        self.last_called = None
        self.cooldown = 0.5
        self.include_from = 0

    def on_created(self, event):
        if event.src_path.endswith(self.raw_data):
            current_time = time.time()
            if self.last_called is None or (current_time - self.last_called) > self.cooldown:
                # print("created",flush=True)
                self.last_called = current_time
                self.handle_raw_data()

    def on_modified(self, event):
        if event.src_path.endswith(self.raw_data):
            current_time = time.time()
            if self.last_called is None or (current_time - self.last_called) > self.cooldown:
                # print("modified",flush=True)
                self.last_called = current_time
                self.handle_raw_data()
            return


    def handle_raw_data(self):
        try:
            self.include_from = create_realtime_bins(neuroncount=20,fname=self.recording_id,include_from =self.include_from)
            self.prepare_realtime_data(self.neuron_data)
            return
        except Exception as e:
            print(f"Error handling file raw data: {e}")    
            traceback.print_exc()

class PredictHandler(FileSystemEventHandler):
    def __init__(self, predict_func,load_func,recording_id ):
        self.recording_id = recording_id
        self.predict = predict_func
        self.lock = threading.Lock()
        self.prepare_realtime_data = load_func

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
    


def run(model_id,model_type,mixed_memory,task,recording_id,parent_connection,iterative_forecast,child_connection,plot_sender,args):
    dataset_data = RealtimeNeuronLaserData(future=args.future,seq_len=args.seq_len,iterative_forecast=iterative_forecast)
    dataset_data.pipe_connection = child_connection
    model = RealtimeForecastModel(model_id = model_id,_data=dataset_data,task=task)
    model.run(recording_id,gpus=args.gpus[0],pipe_connection = parent_connection,plot_sender = plot_sender,model_type=model_type,model_size =args.size,mixed_memory=mixed_memory)


def main_update_plot(plot_listener,fig,axes,lines):
    plot_length,  predict_pos_history,predict_neg_history , recorded_history = plot_listener.recv()
    if plot_length == 0:
        return (fig,axes,lines)  # No data to plot yet

    plot_pos = predict_pos_history.numpy()
    plot_neg = predict_neg_history.numpy()
    history = recorded_history.numpy()
    print("plotting now: ", history.shape, plot_pos.shape,plot_neg.shape,flush=True)
    for i, (line_pos, line_neg, line_rec) in enumerate(lines):
        line_pos.set_data(range(plot_pos.shape[0]), plot_pos[:, i])
        line_neg.set_data(range(plot_neg.shape[0]), plot_neg[:, i])
        line_rec.set_data(range(history.shape[0]), history[:, i])
        # Recalculate limits for the current axis
        # axes[i].relim()
        # axes[i].autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()
    return (fig,axes,lines)

if __name__ == "__main__":
    # https://www.geeksforgeeks.org/dynamically-updating-plot-in-matplotlib/
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
        
    model_id = args.model_id
    recording_id = "dummy"
    print(f" --------- model id: {model_id} recording id : {recording_id}--------- ")
    # Run the main model's run method



    parent_connection, child_connection = Pipe()
    dataset_data = RealtimeNeuronLaserData(future=args.future,seq_len=args.seq_len,iterative_forecast=iterative_forecast)
    dataset_data.set_pipe(child_connection)

    plot_listener, plot_sender = Pipe()

    # fig,axes = plt.subplots(int(dataset_data.out_features/2), 2, figsize=(30,10*dataset_data.out_features),constrained_layout=True)
    # axes = axes.ravel()

    # lines = []
    # for feat_nr, ax in enumerate(axes):
    #     ax.set_ylim(-0.1,None) 
    #     ax.set_xlim(0,64)
    #     line_pos, = ax.plot([], [], lw=1, linestyle="-", alpha =0.5,label="With activation")
    #     line_neg, = ax.plot([], [], lw=1, linestyle="--",alpha =0.5,  label="No activation")
    #     line_rec, = ax.plot([], [], lw=1, linestyle="--", alpha =0.5,label="Recording")
    #     lines.append((line_pos, line_neg, line_rec))
    #     ax.legend(loc='upper right')

    # plt.show(block=False)
    # print("prepared plot")

    processes = [] 
    data_observer = Observer()

    data_observer.daemon = True 
    data_handler = PauseDataHandler(recording_id,dataset_data.prepare_realtime_data)
    data_observer.schedule(data_handler, path=f"recordings/{recording_id}", recursive=False)   
    data_observer.start()

    dummy_data_process = multiprocessing.Process(target=create_dummy_data, args=(recording_id,))    
    dummy_data_process.start()
    processes.append(dummy_data_process)

    predict_process = multiprocessing.Process(target=run,args=(model_id,model_type,mixed_memory,task,recording_id,parent_connection,iterative_forecast,child_connection,plot_sender,args))    
    predict_process.start()
    processes.append(predict_process)



    try:
        while True:
        #     fig,ax, lines = main_update_plot(plot_listener,fig,axes,lines)
        #     plt.pause(0.1)
            pass
    except KeyboardInterrupt:
        plt.close('all')
        data_observer.stop()
        data_observer.join()
        for process in processes:
            process.terminate()

    # plt.close('all')
    data_observer.stop()
    data_observer.join()
    for process in processes:
        process.terminate()

