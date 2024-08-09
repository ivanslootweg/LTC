import numpy as np
import scipy.io
import random
import time
import os
import shutil

def create_dummy_data(recording_id="dummy"):
    mat_contents = scipy.io.loadmat(f"data/neurons/ADL1_2023-10-24_22-40-25.mat",struct_as_record=False, squeeze_me=True) # ADL1_2023-10-24_22-40-25
    mat_contents = {"SessionData": {
                        "CellData": None}
                    }
    n_neurons = 17
    sample_length = 500
    frates = {k : np.random.randint(1,10) for k in range(n_neurons)}
    fake_spikes = list(map(lambda neuron : [], range(n_neurons)))
    time.sleep(5)
    # print("start creating dummy data")
    if not os.path.exists(f"recordings/{recording_id}"):
        os.makedirs(f"recordings/{recording_id}")
    # for t in range(0,10*sample_length,sample_length):
    # since the resolution is in ms we divide each timstep by 1000. 100 times we add a new sequence of dummy data, where each new sequence has a size of 500 ms
    for t in map(lambda x: x/1000.0, range(0,100*sample_length,sample_length)):
        new_fake_spikes = list(map(lambda neuron : list(sorted([
            random.uniform(t,t+sample_length/1000) for i in range(frates[neuron])] ) )  # sample spikes within the time limits of the bin (t,t+sample_length/1000) for i times where i is the firing rate for this neuron in one bin
                , range(n_neurons)))
        fake_spikes =  np.array([list(np.append(fake_spikes[i], new_fake_spikes[i])) for i in range(n_neurons)],dtype=object)
        mat_contents["SessionData"]["CellData"] = fake_spikes
        scipy.io.savemat(f"recordings/{recording_id}/activations_temp.mat",mdict=mat_contents)
        shutil.copy(f"recordings/{recording_id}/activations_temp.mat",f"recordings/{recording_id}/activations.mat")
        time.sleep(2)
    return

if __name__ == "__main:__":
    create_dummy_data()

