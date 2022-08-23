import datalad.api as dl
from eduTools import io
from eduTools import meg as mt
import mne
import mne_bids as mb
import numpy as np
from os import path
import os.path as op
import pandas as pd
import sys

"""
###################
## LOAD SETTINGS ##
###################
"""

# read cfg file
cfg = io.read_json(sys.argv[1])

# paths
bins = cfg['reaction_time_bins']
root = cfg['root']
sess = cfg['sessions']
subs = cfg['subjects']
times = cfg['times']
mode = cfg['mode']
clean_stim = cfg['clean_stim']
clean_resp = cfg['clean_resp']

"""
####################
## CREATE EVOKEDS ##
####################
"""
# create evokeds per session + per coherence level + per response time bin (RT)

for sub in subs:
    for ses in sess:

        # sub-15 only has two sessions
        if sub == 15 and ses == 1:
            continue

        # extract epochs
        ep_name = path.join(root, 'preproc2', f'sub-{sub:02d}',
                            f'ses-{ses:02d}', 'meg',f'sub-{sub:02d}_'\
                            f'ses-{ses:02d}_task-rdm_stim_{mode}_clean-epo.fif')
        epochs = mne.read_epochs(ep_name)

        #extract raw to have access on events
        raw_name = path.join(root, 'preproc2', f'sub-{sub:02d}',
                             f'ses-{ses:02d}', 'meg',f'sub-{sub:02d}_'\
                             f'ses-{ses:02d}_task-rdm_sss_filt-raw_split-01_'\
                             'sss.fif')
        bids_path = mb.BIDSPath(root=op.join(root, 'preproc2', 'inputs'),
                                subject=f'{sub:02d}', session=f'{ses:02d}',
                                task='rdm', datatype='meg')
        epochs_raw = mne.io.read_raw(raw_name)
        events, events_dic = mt.read_events(epochs_raw, bids_path)

        # extract manual bad epochs
        manual_bad = op.join('preproc2', f'sub-{sub:02d}',
                             f'sub-{sub:02d}_{mode}_manual_bad_info.json')
        manual_bad_ses = io.read_json(manual_bad)[f'ses-{ses:02d}']

        # create evoked for one session based on the clean epochs
        epochs.events[manual_bad_ses, 2] = 96
        epochs.event_id['manual_rejection'] = 96
        if mode == "stim":
            ep_clean = epochs.apply_baseline((-0.2,0))
            evoked_ses = ep_clean[clean_stim].average()
        else:
            ep_clean = epochs.apply_baseline((0.1,0.3))
            evoked_ses = ep_clean[clean_resp].average()

        # save evoked of one session
        evoked_ses.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/sub-{sub:02d}_'\
                        f'ses-{ses:02d}_task-rdm_{mode}_clean-epo-ave.fif',
                        overwrite = True)

        # create coherence wise and RT wise evokeds seperatly for stim and resp
        # epochs

        #####################
        ## STIMULUS-LOCKED ##
        #####################

        if mode == "stim":

            # coherence wise evokeds
            evoked_05 = ep_clean['stim_left_0.05', 'stim_right_0.05'].average()
            evoked_1 = ep_clean['stim_left_0.1', 'stim_right_0.1'].average()
            evoked_2 = ep_clean['stim_left_0.2', 'stim_right_0.2'].average()
            evoked_4 = ep_clean['stim_left_0.4', 'stim_right_0.4'].average()

            evoked_05.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.03_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_1.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                          f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                          f'cor-0.06_{mode}_clean-epo-ave.fif',
                          overwrite = True)
            evoked_2.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                          f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                          f'cor-0.12_{mode}_clean-epo-ave.fif',
                          overwrite = True)
            evoked_4.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                          f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                          f'cor-0.24_{mode}_clean-epo-ave.fif',
                          overwrite = True)

            # RT wise evokeds

            # extract RTs
            rt_bins = []
            for epI, ep in enumerate(ep_clean[clean_stim].events):
                idx = np.where(events[:,0] == ep[0])[0]
                resp_idx = idx + 1
                resp_ev = events[resp_idx][0]
                rt = resp_ev[0] - ep[0]
                rt_bins.append([epI,ep[2],rt])

            # dataframe to collect the RTs for each degree of coherence
            rt_bins_05 = pd.DataFrame(columns = ['row_index', 'cor_index',
                                                 'rt'])
            rt_bins_1 = pd.DataFrame(columns = ['row_index', 'cor_index', 'rt'])
            rt_bins_2 = pd.DataFrame(columns = ['row_index', 'cor_index', 'rt'])
            rt_bins_4 = pd.DataFrame(columns = ['row_index', 'cor_index', 'rt'])

            # add values to the dataframe
            for epI, ep in enumerate(rt_bins):
                if ep[1] == 16:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_05 = pd.concat([rt_bins_05, new_row.loc[:]])
                                 .reset_index(drop = True)

                if ep[1] == 32:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_05 = pd.concat([rt_bins_05, new_row.loc[:]])
                                 .reset_index(drop = True)

                if ep[1] == 17:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_1 = pd.concat([rt_bins_1, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 33:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_1 = pd.concat([rt_bins_1, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 18:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_2 = pd.concat([rt_bins_2, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 34:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_2 = pd.concat([rt_bins_2, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 19:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_4 = pd.concat([rt_bins_4, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 35:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_4 = pd.concat([rt_bins_4, new_row.loc[:]])
                                .reset_index(drop = True)

            # convert RTs into float
            rt_bins_05['rt'] = rt_bins_05['rt'].astype(float)
            rt_bins_1['rt'] = rt_bins_1['rt'].astype(float)
            rt_bins_2['rt'] = rt_bins_2['rt'].astype(float)
            rt_bins_4['rt'] = rt_bins_4['rt'].astype(float)

            # split RTs into three bins
            rt_bins_05['RT_bins'] = pd.qcut(rt_bins_05['rt'], q = len(bins),
                                            labels = bins)
            rt_bins_1['RT_bins'] = pd.qcut(rt_bins_1['rt'], q = len(bins),
                                           labels = bins)
            rt_bins_2['RT_bins'] = pd.qcut(rt_bins_2['rt'], q = len(bins),
                                           labels = bins)
            rt_bins_4['RT_bins'] = pd.qcut(rt_bins_4['rt'], q = len(bins),
                                           labels = bins)

            # create RT wise evokeds for each degree of coherence
            for b in bins:
                drop_epochs = []
                for i in range(len(rt_bins_05)):
                    if rt_bins_05.iloc[i]['RT_bins'] != b:
                        drop_epochs.append(i)
                ep_clean_rt = ep_clean['stim_left_0.05', 'stim_right_0.05']
                              .copy().drop(drop_epochs)

                # create evoked for one bin and save it
                evoked_rt_05 = ep_clean_rt.average()
                evoked_rt_05.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                                  f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                                  f'cor-0.03_rt-{b}_{mode}_clean-epo-ave.fif',
                                  overwrite = True)

            for b in bins:
                drop_epochs = []
                for i in range(len(rt_bins_1)):
                    if rt_bins_1.iloc[i]['RT_bins'] != b:
                        drop_epochs.append(i)
                ep_clean_rt = ep_clean['stim_left_0.1', 'stim_right_0.1']
                              .copy().drop(drop_epochs)

                # create evoked for one bin and save it
                evoked_rt_1 = ep_clean_rt.average()
                evoked_rt_1.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                                 f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                                 f'cor-0.06_rt-{b}_{mode}_clean-epo-ave.fif',
                                 overwrite = True)

            for b in bins:
                drop_epochs = []
                for i in range(len(rt_bins_2)):
                    if rt_bins_2.iloc[i]['RT_bins'] != b:
                        drop_epochs.append(i)
                ep_clean_rt = ep_clean['stim_left_0.2', 'stim_right_0.2']
                              .copy().drop(drop_epochs)

                # create evoked for one bin and save it
                evoked_rt_2 = ep_clean_rt.average()
                evoked_rt_2.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                                 f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                                 f'cor-0.12_rt-{b}_{mode}_clean-epo-ave.fif',
                                 overwrite = True)

            for b in bins:
                drop_epochs = []
                for i in range(len(rt_bins_4)):
                    if rt_bins_4.iloc[i]['RT_bins'] != b:
                        drop_epochs.append(i)
                ep_clean_rt = ep_clean['stim_left_0.4', 'stim_right_0.4']
                              .copy().drop(drop_epochs)

                # create evoked for one bin and save it
                evoked_rt_4 = ep_clean_rt.average()
                evoked_rt_4.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                                 f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                                 f'cor-0.24_rt-{b}_{mode}_clean-epo-ave.fif',
                                 overwrite = True)

        #####################
        ## RESPONSE-LOCKED ##
        #####################

        if mode == "resp":

            # coherence wise evokeds
            cor_05 = []
            cor_1 = []
            cor_2 = []
            cor_4 = []
            cor_0 = []

            # extract coherences
            for epI, ep in enumerate(ep_clean[clean_resp].events):
                idx = np.where(events[:,0] == ep[0])[0]
                stim_idx = idx - 1
                stim_ev = events[stim_idx][0][2]
                if stim_ev not in [16, 32]:
                    cor_05.append(epI)
                if stim_ev not in [17, 33]:
                    cor_1.append(epI)
                if stim_ev not in [18, 33]:
                    cor_2.append(epI)
                if stim_ev not in [19, 34]:
                    cor_4.append(epI)

            # create evokeds
            evoked_05 = ep_clean[clean_resp].copy().drop(cor_05).average()
            evoked_1 = ep_clean[clean_resp].copy().drop(cor_1).average()
            evoked_2 = ep_clean[clean_resp].copy().drop(cor_2).average()
            evoked_4 = ep_clean[clean_resp].copy().drop(cor_4).average()

            # save evokeds
            evoked_05.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.03_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_1.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                          f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                          f'cor-0.06_{mode}_clean-epo-ave.fif',
                          overwrite = True)
            evoked_2.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                          f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                          f'cor-0.12_{mode}_clean-epo-ave.fif',
                          overwrite = True)
            evoked_4.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                          f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                          f'cor-0.24_{mode}_clean-epo-ave.fif',
                          overwrite = True)

            # RT wise evokdes

            # extract RTs
            rt_bins = []
            for epI, ep in enumerate(ep_clean[clean_resp].events):
                idx = np.where(events[:,0] == ep[0])[0]
                stim_idx = idx - 1
                stim_ev = events[stim_idx][0]
                stim_cor = events[stim_idx][0][2]
                rt =  ep[0] - stim_ev[0]
                rt_bins.append([epI,stim_cor,rt])


            # dataframe to collect the RTs for each degree of coherence
            rt_bins_05 = pd.DataFrame(columns = ['row_index', 'cor_index',
                                      'rt'])
            rt_bins_1 = pd.DataFrame(columns = ['row_index', 'cor_index', 'rt'])
            rt_bins_2 = pd.DataFrame(columns = ['row_index', 'cor_index', 'rt'])
            rt_bins_4 = pd.DataFrame(columns = ['row_index', 'cor_index', 'rt'])

            # add values to the dataframe
            for epI, ep in enumerate(rt_bins):
                if ep[1] == 16:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_05 = pd.concat([rt_bins_05, new_row.loc[:]])
                                 .reset_index(drop = True)

                if ep[1] == 32:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_05 = pd.concat([rt_bins_05, new_row.loc[:]])
                                 .reset_index(drop = True)

                if ep[1] == 17:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_1 = pd.concat([rt_bins_1, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 33:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_1 = pd.concat([rt_bins_1, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 18:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_2 = pd.concat([rt_bins_2, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 34:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_2 = pd.concat([rt_bins_2, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 19:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_4 = pd.concat([rt_bins_4, new_row.loc[:]])
                                .reset_index(drop = True)

                if ep[1] == 35:
                    new_row = pd.DataFrame({'row_index' : ep[0],
                                            'cor_index' : ep[1], 'rt' : ep[2]},
                                           index = [0])
                    rt_bins_4 = pd.concat([rt_bins_4, new_row.loc[:]])
                                .reset_index(drop = True)

            # convert RTs into float
            rt_bins_05['rt'] = rt_bins_05['rt'].astype(float)
            rt_bins_1['rt'] = rt_bins_1['rt'].astype(float)
            rt_bins_2['rt'] = rt_bins_2['rt'].astype(float)
            rt_bins_4['rt'] = rt_bins_4['rt'].astype(float)

            # split RTs into three bins
            rt_bins_05['RT_bins'] = pd.qcut(rt_bins_05['rt'], q = len(bins),
                                            labels = bins)
            rt_bins_1['RT_bins'] = pd.qcut(rt_bins_1['rt'], q = len(bins),
                                           labels = bins)
            rt_bins_2['RT_bins'] = pd.qcut(rt_bins_2['rt'], q = len(bins),
                                           labels = bins)
            rt_bins_4['RT_bins'] = pd.qcut(rt_bins_4['rt'], q = len(bins),
                                           labels = bins)

            rt_fast_05 = []
            rt_medium_05 = []
            rt_slow_05 = []
            rt_fast_1 = []
            rt_medium_1 = []
            rt_slow_1 = []
            rt_fast_2 = []
            rt_medium_2 = []
            rt_slow_2 = []
            rt_fast_4 = []
            rt_medium_4 = []
            rt_slow_4 = []

            for index, rows in rt_bins_05.iterrows():
                if rows[3] == 'fast':
                    rt_fast_05.append(rows[0])
                if rows[3] == 'medium':
                    rt_medium_05.append(rows[0])
                if rows[3] == 'slow':
                    rt_slow_05.append(rows[0])

            for index, rows in rt_bins_1.iterrows():
                if rows[3] == 'fast':
                    rt_fast_1.append(rows[0])
                if rows[3] == 'medium':
                    rt_medium_1.append(rows[0])
                if rows[3] == 'slow':
                    rt_slow_1.append(rows[0])

            for index, rows in rt_bins_2.iterrows():
                if rows[3] == 'fast':
                    rt_fast_2.append(rows[0])
                if rows[3] == 'medium':
                    rt_medium_2.append(rows[0])
                if rows[3] == 'slow':
                    rt_slow_2.append(rows[0])

            for index, rows in rt_bins_4.iterrows():
                if rows[3] == 'fast':
                    rt_fast_4.append(rows[0])
                if rows[3] == 'medium':
                    rt_medium_4.append(rows[0])
                if rows[3] == 'slow':
                    rt_slow_4.append(rows[0])

            ep_clean = ep_clean[clean_resp]

            # create evokeds and save them
            evoked_05_fast = ep_clean[rt_fast_05].average()
            evoked_05_medium = ep_clean[rt_medium_05].average()
            evoked_05_slow = ep_clean[rt_slow_05].average()
            evoked_1_fast = ep_clean[rt_fast_1].average()
            evoked_1_medium = ep_clean[rt_medium_1].average()
            evoked_1_slow = ep_clean[rt_slow_1].average()
            evoked_2_fast = ep_clean[rt_fast_2].average()
            evoked_2_medium = ep_clean[rt_medium_2].average()
            evoked_2_slow = ep_clean[rt_slow_2].average()
            evoked_4_fast = ep_clean[rt_fast_4].average()
            evoked_4_medium = ep_clean[rt_medium_4].average()
            evoked_4_slow = ep_clean[rt_slow_4].average()

            evoked_05_fast.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.03_rt-fast_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_05_medium.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.03_rt-medium_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_05_slow.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.03_rt-slow_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_1_fast.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.06_rt-fast_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_1_medium.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.06_rt-medium_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_1_slow.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.06_rt-slow_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_2_fast.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.12_rt-fast_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_2_medium.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.12_rt-medium_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_2_slow.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.12_rt-slow_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_4_fast.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.24_rt-fast_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_4_medium.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.24_rt-medium_{mode}_clean-epo-ave.fif',
                           overwrite = True)
            evoked_4_slow.save(f'analysed/sub-{sub:02d}/ses-{ses:02d}/'\
                           f'sub-{sub:02d}_ses-{ses:02d}_task-rdm_'\
                           f'cor-0.24_rt-slow_{mode}_clean-epo-ave.fif',
                           overwrite = True)
