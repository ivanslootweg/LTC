import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import argparse
import shutil

# Data loading
def firing_rate(binary_spikes,bin_centers,bin_width):
    # calculate firing_rate within a bin. bin is defined by its center and width
    # https://stackoverflow.com/questions/57631469/extending-histogram-function-to-overlapping-bins-and-bins-with-arbitrary-gap
    # https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html 
    idx1 = np.searchsorted(binary_spikes,bin_centers-bin_width,'right')
    idx2 = np.searchsorted(binary_spikes,bin_centers+bin_width,'left')
    rate = idx2-idx1
    return rate

def left_smoothing(signal, window_type="hamming", window_length=15):
    window = scipy.signal.get_window(window_type, window_length) / np.sum(window_length)
    filtered_signal = scipy.signal.convolve(signal, window, mode="same")
    return filtered_signal

def create_spikes(spikes,bin_distance = None,timepoints_flat=None,bin_width= None,overlap =0, plot=True, save = True,sigma=None,hamming = False, is_pulses=False):
    bin_centers_neurons = np.arange(np.around(min(timepoints_flat),decimals = 2), np.around(max(timepoints_flat),decimals = 2),bin_distance) # 1/10 second resolution
    firing_rates = []
    if overlap is not None and bin_width is not None :
        raise ValueError("cannot set both binwidth and overlap. Choose 1")
    if overlap is not None:
        bin_width = bin_distance + overlap
    elif bin_width is not None:
        overlap = bin_width - bin_distance


    for i in range(spikes.shape[0]):
        firing_rates.append(firing_rate(spikes[i],bin_centers_neurons,bin_width).astype(np.float32))
    firing_rates = np.array(firing_rates)
    if sigma and not is_pulses:
        firing_rates = gaussian_filter1d(firing_rates,sigma)
        smoothing_tag = f"_s{sigma}"
    elif hamming and not is_pulses:
        for i in range(firing_rates.shape[0]):
            firing_rates[i] = left_smoothing(firing_rates[i])
        smoothing_tag = f"_hamming"
    else:
        smoothing_tag = ""
    # if plot:
    #     fig, axes = plt.subplots(spikes.shape[0], 1, figsize=(30,4*spikes.shape[0]),constrained_layout=True)
    #     if not is_pulses:
    #         axes = axes.ravel()
    #         for (feat_nr,ax) in zip(range(spikes.shape[0]),axes):
    #             ax.plot(firing_rates[feat_nr],label="Binned activation",linewidth=1)
    #             ax.set_title(f"feat {feat_nr}",loc = 'left')
    #             ax.legend(loc='upper right')

    #         # plt.tight_layout()
    #         plt.suptitle(f"bin width {bin_width:4f}, distance {bin_distance:2f}")
    #         if save:
    #             plt.savefig(f"data/binned_f{bin_distance:2f}_w{bin_width:4f}_n{N_NEURONS}{smoothing_tag}.jpg")
    #     else:
    #         fig = plt.plot(firing_rates[0],label = "binned laser pulses",linewidth=1)
    #         plt.title("laser pulses")
    #         plt.legend(loc="upper right")
    #     plt.show()
    #     plt.close()

    return firing_rates


def load_training_data(neuroncount=None,fname=None):
    N_NEURONS = {"ADL1_2023-10-24_22-40-25" : 17,
                 "ADL1_2023-07-31_00-09-22" : 20
                 }
    if not neuroncount:
        neuroncount = N_NEURONS[fname]
    mat_contents = scipy.io.loadmat(f"data/neurons/{fname}",struct_as_record=False, squeeze_me=True)
    raw_spikes = mat_contents["SessionData"].CellData
    raw_laserpulses = mat_contents["SessionData"].Laser_infor

    if fname == "ADL1_2023-10-24_22-40-25" :
        raw_spikes = np.delete(raw_spikes,8,0)

    # remove trailing timeseries
    raw_timepoints_flat = np.array([])
    for i in range(len(raw_spikes)):
        raw_timepoints_flat = np.append(raw_timepoints_flat, raw_spikes[i].flatten())
    raw_timepoints_flat = np.append(raw_timepoints_flat,raw_laserpulses)
    timepoints_flat = raw_timepoints_flat - min(raw_timepoints_flat)
    spikes = np.array(list(map(lambda x: x - min(raw_timepoints_flat), raw_spikes)), dtype=object)
    laserpulses = np.array(list(map(lambda x: x - min(raw_timepoints_flat), raw_laserpulses)), dtype=object)
    T = max(timepoints_flat)

    with open(f"data/neurons/data_description_{fname}.txt", "w") as f:
        f.write("neuron spikes: \n")
        for neuron in spikes:
            f.write(f"{min(neuron)} {max(neuron)} firing rate {len(neuron) / T} Hz. some inter-spike times: {[neuron[i+1]-neuron[i] for i in range(0,20,5)]}\n")
        freqs = [len(n)/ T for n in spikes]
        f.write(f"Firing rates min: {np.min(freqs)} max: {np.max(freqs)} mean: {np.mean(freqs)}\n")
        f.write("laser spikes: \n")
        f.write(f"{min(laserpulses)} {max(laserpulses)} some inter-pulse times: {[laserpulses[i+1]-laserpulses[i] for i in range(0,20,5)]}\n")
    f.close()


    bin_distance = 0.05
    overlap = 0.025
    sigma  = 5
    firing_rates = create_spikes(spikes,bin_distance=bin_distance,overlap = overlap,timepoints_flat=timepoints_flat,plot=False,sigma=sigma)
    np.save(f"data/neurons/activations_{fname}.npy",firing_rates)
    pulse_rates = create_spikes(np.expand_dims(laserpulses,0),bin_distance=bin_distance,overlap = overlap,timepoints_flat=timepoints_flat,plot=False,is_pulses=True)
    np.save(f"data/neurons/laserpulses_{fname}.npy",pulse_rates)


def create_realtime_spikes(spikes,bin_distance = None,timepoints_flat=None,bin_width= None,overlap =0, save = True,sigma=None,hamming = False, is_pulses=False):
    bin_centers_neurons = np.arange(np.around(min(timepoints_flat),decimals = 2), np.around(max(timepoints_flat),decimals = 2),bin_distance) # 1/10 second resolution
    firing_rates = []
    if overlap is not None and bin_width is not None :
        raise ValueError("cannot set both binwidth and overlap. Choose 1")
    if overlap is not None:
        bin_width = bin_distance + overlap
    elif bin_width is not None:
        overlap = bin_width - bin_distance

    for i in range(spikes.shape[0]):
        firing_rates.append(firing_rate(spikes[i],bin_centers_neurons,bin_width).astype(np.float32))
    firing_rates = np.array(firing_rates)
    if sigma and not is_pulses:
        firing_rates = gaussian_filter1d(firing_rates,5)
    elif hamming and not is_pulses:
        for i in range(firing_rates.shape[0]):
            firing_rates[i] = left_smoothing(firing_rates[i])

    return firing_rates


def create_realtime_bins(neuroncount=None,fname=None):
    # print("start binning")
    N_NEURONS = {"ADL1_2023-10-24_22-40-25" : 17,
                 "ADL1_2023-07-31_00-09-22" : 20
                 }
    if not neuroncount:
        neuroncount = N_NEURONS[fname]
    mat_contents = scipy.io.loadmat(f"recordings/{fname}/activations.mat",struct_as_record=False, squeeze_me=True)
    raw_spikes = mat_contents["SessionData"].CellData
    # remove trailing timeseries
    raw_timepoints_flat = np.array([])
    for i in range(len(raw_spikes)):
        raw_timepoints_flat = np.append(raw_timepoints_flat, raw_spikes[i].flatten())
    timepoints_flat = raw_timepoints_flat - min(raw_timepoints_flat)
    spikes = np.array(list(map(lambda x: x - min(raw_timepoints_flat), raw_spikes)), dtype=object)
    T = max(timepoints_flat)

    bin_distance = 0.05
    overlap = 0.025
    sigma  = 5
    firing_rates = create_realtime_spikes(spikes,bin_distance=bin_distance,overlap = overlap,timepoints_flat=timepoints_flat,sigma=sigma)
    np.save(f"recordings/{fname}/activations.npy",firing_rates)
    # print("done with bins", firing_rates.shape,firing_rates[0][:-15])
    # print("done binning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname',default="ADL1_2023-10-24_22-40-25")
    parser.add_argument('--neuroncount',default=None,type=int)
    args = parser.parse_args()
    load_training_data(args.neuroncount,args.fname)
    

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fname',default="ADL1_2023-10-24_22-40-25")
    # parser.add_argument('--neuroncount',default=None,type=int)
    # args = parser.parse_args()
    # create_realtime_bins(args.neuroncount,args.fname)
    