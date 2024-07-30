import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import argparse
import shutil
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Data loading
def calculate_firing_rate(binary_spikes,bin_centers,interval):
    # calculate firing_rate within a bin. bin is defined by its center and width
    # https://stackoverflow.com/questions/57631469/extending-histogram-function-to-overlapping-bins-and-bins-with-arbitrary-gap
    # https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html 
    # print(bin_centers-interval)
    idx1 = np.searchsorted(np.atleast_1d(binary_spikes),bin_centers-interval,'right')
    idx2 = np.searchsorted(np.atleast_1d(binary_spikes),bin_centers+interval,'left')
    rate = idx2-idx1
    return rate

def left_smoothing(signal, window_type="hamming", window_length=15):
    window = scipy.signal.get_window(window_type, window_length) / np.sum(window_length)
    filtered_signal = scipy.signal.convolve(signal, window, mode="same")
    return filtered_signal

def create_spikes(spikes,bin_distance = None,include_from=None,include_until=None, plot=False, overlap =0, sigma=None,hamming = False, is_pulses=False):
    bin_centers_neurons = np.arange(np.around(include_from,decimals = 4), np.around(include_until,decimals = 4),bin_distance)
    firing_rates = []

    if overlap is not None:
        interval = (bin_distance-overlap)/2 + overlap
    else:
        interval = bin_distance/2 



    for i in range(spikes.shape[0]):
        firing_rates.append(calculate_firing_rate(spikes[i],bin_centers_neurons,interval).astype(np.float32))
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

    if plot:

        x1 = 15000
        x2 = 15500

        extent = (0,len(firing_rates[0])
        Z2 = np.zeros()

        fig, axes = plt.subplots(spikes.shape[0], 1, figsize=(30,4*spikes.shape[0]),constrained_layout=True)
        axes = axes.ravel()
        for (feat_nr,ax) in zip(range(spikes.shape[0]),axes):
            ax.plot(firing_rates[feat_nr],label="Binned activation",linewidth=1)
            ax.set_title(f"feat {feat_nr}",loc = 'left')
            ax.legend(loc='upper right')
            # select y-range for zoomed region
            y1 = min(firing_rates[feat_nr])
            y2 = max(firing_rates[feat_nr])
            axins = zoomed_inset_axes(ax, 1, loc=1) # zoom = 2
            axins.plot(firing_rates[feat_nr])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            plt.draw()


# prepare the demo image
Z, extent = get_demo_image()
Z2 = np.zeros([150, 150], dtype="d")
ny, nx = Z.shape
Z2[30:30 + ny, 30:30 + nx] = Z

# extent = [-3, 4, -4, 3]
ax.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")

axins = zoomed_inset_axes(ax, 6, loc=1)  # zoom = 6
axins.imshow(Z2, extent=extent, interpolation="nearest",
             origin="lower")

# sub region of the original image
x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.draw()


        # plt.tight_layout()
        plt.suptitle(f"bin width {str(interval)}, distance {str(bin_distance)}")
        plt.savefig(f"data/neurons/binned_width{interval:.2f}_distance{bin_distance:.4f}.jpg")
        plt.close()


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
        raw_timepoints_flat = np.append(raw_timepoints_flat, [raw_spikes[i]])
    raw_timepoints_flat = np.append(raw_timepoints_flat,raw_laserpulses)
    timepoints_flat = raw_timepoints_flat - min(raw_timepoints_flat)
    spikes = np.array(list(map(lambda x: x - min(raw_timepoints_flat), raw_spikes)), dtype=object)
    laserpulses = np.array(list(map(lambda x: x - min(raw_timepoints_flat), raw_laserpulses)), dtype=object)
    T = max(timepoints_flat)


    bin_distance = 0.05
    overlap = 0.025
    sigma  = 8
    firing_rates = create_spikes(spikes,include_from=min(timepoints_flat),include_until=max(timepoints_flat), plot=True,
        bin_distance=bin_distance,overlap = overlap,sigma=sigma)
    np.save(f"data/neurons/activations_{fname}.npy",firing_rates)
    pulse_rates = create_spikes(np.expand_dims(laserpulses,0),include_from=min(timepoints_flat),include_until=max(timepoints_flat), plot=False,
        bin_distance=bin_distance,overlap = overlap,sigma=sigma,is_pulses =True)
    np.save(f"data/neurons/laserpulses_{fname}.npy",pulse_rates)

    with open(f"data/neurons/data_description_{fname}.txt", "w") as f:
        f.write(f"binned with bin distance {bin_distance} overlap {overlap} sigma {sigma} \n\n")
        f.write("neuron spikes: \n")
        for neuron in spikes:
            f.write(f"{min(neuron)} {max(neuron)} firing rate {len(neuron) / T} Hz. some inter-spike times: {[neuron[i+1]-neuron[i] for i in range(0,20,5)]}\n")
        freqs = [len(n)/ T for n in spikes]
        f.write(f"Firing rates min: {np.min(freqs)} max: {np.max(freqs)} mean: {np.mean(freqs)}\n")
        f.write("laser spikes: \n")
        f.write(f"{min(laserpulses)} {max(laserpulses)} some inter-pulse times: {[laserpulses[i+1]-laserpulses[i] for i in range(0,20,5)]}\n")
    f.close()



def create_realtime_spikes(spikes,bin_distance = None,include_from=None,include_until=None, overlap =0, save = True,sigma=None,hamming = False, is_pulses=False):
    # remove last bin 
    bin_centers_neurons = np.arange(np.around(include_from,decimals = 4), np.around(include_until,decimals = 4),bin_distance)[:-1]

    firing_rates = []
    if overlap is not None:
        interval = (bin_distance-overlap)/2 + overlap
    else:
        interval = bin_distance/2 


    for i in range(spikes.shape[0]):
        firing_rates.append(calculate_firing_rate(spikes[i],bin_centers_neurons,interval).astype(np.float32))
    firing_rates = np.array(firing_rates)
    if sigma and not is_pulses:
        firing_rates = gaussian_filter1d(firing_rates,sigma)
    elif hamming and not is_pulses:
        for i in range(firing_rates.shape[0]):
            firing_rates[i] = left_smoothing(firing_rates[i])

    return firing_rates,bin_centers_neurons[-1]


def create_realtime_bins(neuroncount=None,fname=None,include_from=0):
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
        raw_timepoints_flat = np.append(raw_timepoints_flat, [raw_spikes[i]])
    # timepoints_flat = raw_timepoints_flat - min(raw_timepoints_flat)
    # spikes = np.array(list(map(lambda x: x - min(raw_timepoints_flat), raw_spikes)), dtype=object)
    spikes = np.array(list(map(lambda x: x, raw_spikes)), dtype=object)
    timepoints_flat = raw_timepoints_flat
    T = max(timepoints_flat)
    bin_distance = 0.05
    overlap = 0.025
    sigma  = 5
    firing_rates,last_bin_center = create_realtime_spikes(spikes,include_from=include_from,include_until=max(timepoints_flat),
        bin_distance=bin_distance,overlap = overlap,sigma=sigma)
    # print(f"spikes max {T}, spikes min {min(timepoints_flat)} total bins: {firing_rates.shape}",flush=True)
    np.save(f"recordings/{fname}/activations.npy",firing_rates)
    return (last_bin_center + (bin_distance+overlap)/2 - overlap) # beginning of the next bin
    # print("done with bins", firing_rates.shape,firing_rates[0][:-15])
    # print("done binning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname',default="ADL1_2023-07-31_00-09-22")
    parser.add_argument('--neuroncount',default=None,type=int)
    args = parser.parse_args()
    load_training_data(args.neuroncount,args.fname)
    

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fname',default="ADL1_2023-10-24_22-40-25")
    # parser.add_argument('--neuroncount',default=None,type=int)
    # args = parser.parse_args()
    # create_realtime_bins(args.neuroncount,args.fname)
    