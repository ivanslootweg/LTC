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
    sample_length = 32
    frates = {k : 1-np.abs(random.uniform(-0.2,0.8))  for k in range(n_neurons)}
    fake_spikes = list(map(lambda neuron : [random.uniform(0,sample_length) for i in range(int(frates[neuron] * sample_length))], range(n_neurons)))
    time.sleep(20)
    # print("start creating dummy data")
    if not os.path.exists(f"recordings/{recording_id}"):
        os.makedirs(f"recordings/{recording_id}")
    for t in range(sample_length,10000,sample_length):
        # print(fake_spikes.shape,fakqe_spikes[0])
        new_fake_spikes = list(map(lambda neuron : [random.uniform(t,t+sample_length) for i in range(int(frates[neuron] * sample_length))], range(n_neurons)))
        fake_spikes =  np.array([np.append(fake_spikes[i], new_fake_spikes[i]).tolist() for i in range(n_neurons)],dtype=object)
        mat_contents["SessionData"]["CellData"] = fake_spikes
        # scipy.io.savemat(f"recordings/{recording_id}/activations2.mat",mdict=mat_contents)
        # saved_mat_contents = scipy.io.loadmat(f"recordings/{recording_id}/activations2.mat",struct_as_record=False, squeeze_me=True)
        print("saving .mat..")
        scipy.io.savemat(f"recordings/{recording_id}/activations_temp.mat",mdict=mat_contents)
        print("saved .mat..")
        shutil.copy(f"recordings/{recording_id}/activations_temp.mat",f"recordings/{recording_id}/activations.mat")
        print("renamed")
        time.sleep(100)



if __name__ == "__main:__":
    create_dummy_data()

