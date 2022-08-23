import ioUtils as io
import mne
from mne.stats import permutation_t_test
import numpy as np
from sys import argv

"""
###################
## LOAD SETTINGS ##
###################
"""

# read cfg file
cfg = io.read_write_json(argv[1])

# paths
bins = cfg['reaction_time_bins']
cors = cfg['coherences']
mode = cfg['mode']
polarity = cfg['polarity']
sensors = cfg['sensors']
sess = cfg['sessions']
subs = cfg['subjects']

"""
############################
## CREATE SUBJECT EVOKEDS ##
############################
"""

# create evokeds per subject + grand average

# list to collect subject evokeds to average into a grand average
all_subs = []

# array to collect subject evokeds to run later permutations on
if mode == 'stim':
    all_evok = np.zeros((23,306,3001))
else:
    all_evok = np.zeros((23,306,3201))

for subI, sub in enumerate(subs):

    # loop over subjects to get subjectwise evokeds
    for ses in sess:

        # sub 15 only has 2 sessions
        if sub == 15 and ses == 1:
            continue

        # list to collect session evokeds to average into subject evoked
        ses_ev = []

        # add one session evoked
        ses_ev.append(mne.read_evokeds(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                      f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_{mode}_'\
                      'clean-epo-ave.fif')[0])

    # average over the three sessions and save it
    sub_ev  = mne.grand_average(ses_ev)
    sub_ev.save(f'analysed/sub-{sub:02d}/sub-{sub:02d}_task-rdm_{mode}_'\
                'clean-epo-ave.fif', overwrite = True)

    # add one subject evoked
    all_subs.append(sub_ev)
    all_evok[subI,:]= sub_ev.get_data()

    # plot subject to get an overview
    if mode == 'stim':
        times = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.2]
    else:
        times = [ -2.0, -1.7, -1.4, -1.1, -0.8, -0.5, -0.2, 0.1, 0.4]
    fig = sub_ev.plot_topomap(colorbar = True,times = times, show = False)
    io.savePlot(fig, outpath = f'analysed/sub-{sub:02d}/sub-{sub:02d}_'\
                f'task-rdm_{mode}_overview_clean-epo-ave.png', dpi = 600)

# average over all subjects (= grand average) and save it
grand_average = mne.grand_average(all_subs)
grand_average.save(f'analysed/task-rdm_{mode}_clean-epo-ave.fif',
                   overwrite = True)

# plot grand grand_average to get an overview
fig = grand_average.plot_topomap(colorbar = True,times = times, show = False)
io.savePlot(fig, outpath = f'analysed/task-rdm_{mode}_overview_'\
            'clean-epo-ave.png', dpi = 600)

"""
##########################
## SIGNIFICANCE TESTING ##
##########################
"""

# run permutation t-test on sensor data to get CPP sensors

# pick all sensors (magnetometers and gradiometers)
picks = mne.pick_types(grand_average.info, meg = True, exclude ='bads')

# extract time points from the grand average
time_p = grand_average.times

# create temporal mask according to the mean response time
if mode == "stim":
    temporal_mask = np.logical_and(0.9 <= time_p, time_p <= 1.5)
if mode == "resp":
    temporal_mask = np.logical_and(-0.3 <= time_p, time_p <= 0.3)

all_evok = np.mean(all_evok[:, :, temporal_mask], axis = 2)

# run permutaton t-test to test sensors acitivity against null-distribution
n_permutations = 50000
T0, p_values, H0 = permutation_t_test(X = all_evok,
                                      n_permutations = n_permutations,
                                      n_jobs = 1)

# extract significant sensors
sig_sensors = picks[p_values <= 0.05]
sig_sensors_names = [grand_average.ch_names[k] for k in sig_sensors]
print("Number of significant sensors : %d" % len(sig_sensors))
print("Sensors names : %s" % sig_sensors_names)

# create grand_average of only the significant sensors
sig_grand_average = grand_average.copy().pick_channels(sig_sensors_names)

# split the two sensor clusters and do analysis seperatly
if mode == 'resp' and sensors == 'mag':
    sig_mag_pos = []
    sig_mag_neg = []
    idxt = next(x for x, val in enumerate(sig_grand_average.times) if val > 0.2)
    for i in range(len(sig_grand_average.data)):
        if sig_grand_average.data[i,idxt] > 0:
            sig_mag_pos.append(sig_grand_average.ch_names[i])
        if sig_grand_average.data[i,idxt] < 0:
            sig_mag_neg.append(sig_grand_average.ch_names[i])
    if polarity == 'pos':
        sig_grand_average = grand_average.copy().pick_channels(sig_mag_pos)
                            .pick_types(sensors)
    if polarity == 'neg':
        sig_grand_average = grand_average.copy().pick_channels(sig_mag_neg)
                            .pick_types(sensors)

# visualize in topoplot with p-values
mask_grand_average = mne.EvokedArray(-np.log10(p_values)[:, np.newaxis],
                                     grand_average.info, tmin = 0.)
stats_picks = mne.pick_channels(mask_grand_average.ch_names, sig_sensors_names)
mask = p_values[:, np.newaxis] <= 0.05
fig = mask_grand_average.plot_topomap(times = [0], scalings = 1,
                                      time_format = None, cmap = 'Reds',
                                      vmin = 0., vmax = np.max,
                                      units = '-log10(p)', cbar_fmt = '-%0.1f',
                                      mask = mask, size = 3,
                                      show_names = lambda x: x[4:] + ' ' * 20,
                                      time_unit = 's', ch_type = sensors,
                                      show = False)
io.savePlot(fig, outpath = 'analysed/p-value_significant-sensors_'
            f'{sensors}_{mode}.png', dpi = 600)

# visualize in topoplot with activity
if mode == "stim":
    sig_times = (1,1.2,1.4)
if mode == "resp":
    sig_times = (-0.2,0,0.2)
_times = ((np.abs(grand_average.times - t)).argmin() for t in sig_times)
sig_list = [tuple(sig_sensors_names), tuple(sig_sensors_names),
            tuple(sig_sensors_names)]
_sensors = [np.in1d(grand_average.ch_names, sen) for sen in sig_list]
mask = np.zeros(grand_average.data.shape, dtype = 'bool')
for _sens, _time in zip(_sensors, _times):
    mask[_sens, _time] = True
fig = grand_average.plot_topomap(sig_times, ch_type = sensors, time_unit = 's',
                                 mask = mask,
                                 mask_params = dict(markersize = 10),
                                 show = True)
io.savePlot(fig, outpath = f'analysed/significant-sensors_{sensors}_{mode}.png',
            dpi = 600)


# low-pass filter 10 Hz the grand average to plot the Root Mean Square (RMS)
lpf_sig_grand_average = sig_grand_average.copy().filter(l_freq = None,
                                                        h_freq = 10)
fig = lpf_sig_grand_average.plot(gfp = 'only', picks = sensors, show = False)
io.savePlot(fig, outpath = f'analysed/significant-grand_average_{mode}_'\
            f'{sensors}_{polarity}.png', dpi = 600)
io.savePlot(fig, outpath = f'analysed/significant-grand_average_{mode}_'\
            f'{sensors}_{polarity}.svg', dpi = 600)
