import functions as fc
from ioUtils import read_write_json
import mne
import pandas as pd
import pingouin as pg
from significance_testing import sig_sensors_names
from sys import argv

"""
###################
## LOAD SETTINGS ##
###################
"""

# read cfg files
cfg_settings = read_write_json(argv[1])
cfg_run = read_write_json(argv[-1])

# paths
bins = cfg_settings['reaction_time_bins']
cors = cfg_settings['coherences']
mode = cfg_settings['mode']
polarity = cfg_settings['polarity']
sensors = cfg_settings['sensors']
sess = cfg_settings['sessions']
subs = cfg_settings['subjects']

if mode == 'resp' and sensors == 'mag':
    from significance_testing import sig_mag_pos, sig_mag_neg

"""
###########################
## INTEGRATION PRINCIPLE ##
###########################
"""

if cfg_run['integration_principle'] == 'run':

    # hyphtesis: the buildup rate of the CPP increases in proportion with the
    #            strength of the coherent motion

    # dataframe to collect slope per subject per coherence level per subject
    slopes = pd.DataFrame(columns = ['subject', 'coherence', 'slope'])

    for sub in subs:

        # loop over coherence level to get levelwise evokeds
        for cor in cors:

            # list to collect session evokeds to average into coherence evoked
            ses_ev = []

            # loop over the three sessions to get coherencewise evokeds
            for ses in sess:

                # sub 15 only has 2 sessions
                if sub == 15 and ses == 1:
                    continue
                # add one session evoked
                ses_ev.append(mne.read_evokeds(f'analysed/sub-{sub:02d}/'\
                              f'ses-{ses:02d}/sub-{sub:02d}_ses-{ses:02d}_'\
                              f'task-rdm_cor-{cor}_{mode}_clean-epo-ave.fif')
                              [0])

            # average over the three sessions and save it
            cor_ev = mne.grand_average(ses_ev)
            cor_ev.save(f'analysed/sub-{sub:02d}/sub-{sub:02d}_task-rdm_'\
                        f'cor-{cor}_{mode}_clean-epo-ave.fif', overwrite = True)

            # pick coherence evoked only for the significant sensors
            # and one sensors type
            if mode == 'resp' and sensors == 'mag':

                # split the two sensor clusters and do analysis seperatly
                if polarity == 'pos':
                    sig_cor_ev = cor_ev.copy().pick_channels(sig_mag_pos)
                if polarity == 'neg':
                    sig_cor_ev = cor_ev.copy().pick_channels(sig_mag_neg)
            else:
                sig_cor_ev = cor_ev.copy().pick_channels(sig_sensors_names)

            sig_cor_ev  = sig_cor_ev.pick_types(sensors)
            sig_cor_ev.save(f'analysed/sub-{sub:02d}/sub-{sub:02d}_task-rdm_'\
                            f'cor-{cor}_{mode}_{sensors}_{polarity}_signicant_'\
                            'clean-epo-ave.fif', overwrite = True)

            # find coordinates for slope
            if mode == 'stim':
                coord1, coord2 = fc.coordinates(sig_cor_ev,time1 = 0.2,
                                                time2 =  0.35)
            if mode == 'resp':
                coord1, coord2 = fc.coordinates(sig_cor_ev,time1 = -0.25,
                                                time2 = -0.10)

            # calculate slope
            slope_cor = fc.slope_calc(coord1[0],coord1[1],coord2[0],coord2[1])

            # append results to dataframe
            new_row = pd.DataFrame({'subject': sub, 'coherence': cor,
                                    'slope': slope_cor}, index = [0])
            slopes = pd.concat([slopes, new_row.loc[:]])

    # one-way repeated measures ANOVA
    # test if the slope significantly varies with the coherence level
    print(pg.rm_anova(dv = 'slope', within = 'coherence', subject = 'subject',
                      data = slopes))

    # add t-tests if significant
    print(pg.pairwise_tests(dv= 'slope', within= 'coherence',
          subject= 'subject', data= slopes))

else:
    print("The integration principle is not being tested.")

"""
##############################
## PREDICTIVE BUILD-UP RATE ##
##############################
"""

if cfg_run['predictive_build-up_rate'] == 'run':

    # hypothesis: steeper build-up rate is associated with faster response times
    #             (RTs)

    # dataframe to collect slope per RT bin per coherence level
    slopes = pd.DataFrame(columns = ['subject','coherence', 'bins', 'slope'])

    for cor in cors:

        for b in bins:

            # loop over subjects to get binwise evokeds per subject for
            # each coherence level
            for sub in subs:

                # list to collect session evokeds to average into RT evoked
                ses_ev = []

                for ses in sess:

                    # sub 15 only has 2 sessions
                    if sub == 15 and ses == 1:
                        continue

                    # add one session evoked
                    ses_ev.append(mne.read_evokeds(f'analysed/sub-{sub:02d}/'\
                                                   f'ses-{ses:02d}/'\
                                                   f'sub-{sub:02d}_'\
                                                   f'ses-{ses:02d}_task-rdm_'\
                                                   f'cor-{cor}_rt-{b}_{mode}_'\
                                                   'clean-epo-ave.fif')[0])

                # average over the three sessions and save it
                sub_ev = mne.grand_average(ses_ev)
                sub_ev.save(f'analysed/sub-{sub:02d}/sub-{sub:02d}_task-rdm_'\
                            f'cor-{cor}_rt-{b}_{mode}_clean-epo-ave.fif',
                            overwrite = True)

                # pick RT evoked only for the significant sensors
                # and one sensors type
                if mode == 'resp' and sensors == 'mag':

                    # split the two sensor clusters and do analysis seperatly
                    if polarity == 'pos':
                        sig_sub_ev = sub_ev.copy().pick_channels(sig_mag_pos)
                    if polarity == 'neg':
                        sig_sub_ev = sub_ev.copy().pick_channels(sig_mag_neg)
                else:
                    sig_sub_ev = sub_ev.pick_channels(sig_sensors_names)

                sig_sub_ev  = sig_sub_ev.pick_types(sensors)
                sig_sub_ev.save(f'analysed/task-rdm_cor-{cor}_rt-{b}_'\
                                f'significant_{mode}_{sensors}_{polarity}_'\
                                'clean-epo-ave.fif', overwrite = True)

                # find coordinates for line
                if mode == 'stim':
                    coord1, coord2 = fc.coordinates(sig_sub_ev,time1 = 0.2,
                                                    time2 = 0.35)
                if mode == 'resp':
                    coord1, coord2 = fc.coordinates(sig_sub_ev,time1 = -0.25,
                                                    time2 = -0.10)

                # calculate slope
                slope_cor = fc.slope_calc(coord1[0], coord1[1],
                                          coord2[0], coord2[1])

                # append results to dataframe
                new_row = pd.DataFrame({'subject': sub, 'coherence': cor,
                                        'bins': b, 'slope': slope_cor},
                                       index = [0])
                slopes = pd.concat([slopes, new_row.loc[:]])

    # two-way repeated measures ANOVA
    # test if the slope significantly varies with RT bin and coherence level
    print(pg.rm_anova(dv = 'slope', within = ['coherence', 'bins'],
                      subject = 'subject', data = slopes))

    # add post-hoc tests if significant
    print(pg.pairwise_tests(dv='slope', within=['coherence', 'bins'],
                            subject='subject', data=slopes))

else:
    print("The predicte build-up rate is not being tested.")

"""
#################################
## ACTION-TRIGGERING THRESHOLD ##
#################################
"""

if cfg_run['action-triggering_threshold'] == 'run':

    # hypothesis: action-triggering threshold is independent of RT

    # dataframe to collect peak per RT bin per subject
    peaks = pd.DataFrame(columns = ['subject', 'bins', 'peak'])

    for b in bins:

        # loop over subjects to get binwise evokeds per subject
        for sub in subs:

            # variable to create evoked over all coherence levels
            cor_ev = []

            for cor in cors:

                # list to collect session evokeds to average into
                # coherence evoked
                ses_ev = []

                for ses in sess:

                    # sub 15 only has 2 sessions
                    if sub == 15 and ses == 1:
                        continue

                    # add one session evoked
                    ses_ev.append(mne.read_evokeds(f'analysed/sub-{sub:02d}/'\
                                                   f'ses-{ses:02d}/'\
                                                   f'sub-{sub:02d}_'\
                                                   f'ses-{ses:02d}_task-rdm_'\
                                                   f'cor-{cor}_rt-{b}_{mode}_clean-'\
                                                   'epo-ave.fif')[0])

                # average over the three sessions
                cor_ev.append(mne.grand_average(ses_ev))

            # average over the coherence levels
            sub_ev = mne.grand_average(cor_ev)
            sub_ev.save(f'analysed/sub-{sub:02d}/sub-{sub:02d}_task-rdm_'\
                        f'rt-{b}_{mode}_clean-epo-ave.fif', overwrite = True)

            # pick RT evoked only for the significant sensors
            # and one sensors type

            if mode == 'resp' and sensors == 'mag':

                #split the two sensor clusters and do analysis seperatly
                if polarity == 'pos':
                    sig_sub_ev = sub_ev.copy().pick_channels(sig_mag_pos)
                if polarity == 'neg':
                    sig_sub_ev = sub_ev.copy().pick_channels(sig_mag_neg)
            else:
                sig_sub_ev = sub_ev.pick_channels(sig_sensors_names)

            sig_sub_ev  = sig_sub_ev.pick_types(sensors)
            sig_sub_ev.save(f'analysed/task-rdm_rt-{b}_{mode}_{sensors}_'\
                            f'{polarity}_significant_clean-epo-ave.fif',
                            overwrite = True)

            # define peak for each sub evoked
            if mode == "stim":
                peak = fc.find_peak(sig_sub_ev, mode = mode, start = 1.0,
                                    end = 1.5)
            if mode == "resp":
                peak = fc.find_peak(sig_sub_ev, mode = mode, start = -0.25,
                                    end = 0.25)

            # append results to dataframe
            new_row = pd.DataFrame({'subject': sub, 'bins': b, 'peak': peak},
                                   index = [0])
            peaks = pd.concat([peaks, new_row.loc[:]])

    # one-way repeated measures ANOVA
    # test if the peak significantly varies with RT bin
    print(pg.rm_anova(dv = 'peak', within = 'bins', subject = 'subject',
                      data = peaks))

    # need Bayes statistic --> switch to JASP
    peaks_jasp = pd.DataFrame(columns = ['subject', 'peak_slow', 'peak_medium',
                                         'peak_fast'])
    for sub in subs:
        rows_of_interest = peaks.loc[peaks['subject'] == sub]
        new_row = pd.DataFrame({'subject': sub,
                                'peak_slow': rows_of_interest.iloc[0,2],
                                'peak_medium': rows_of_interest.iloc[1,2],
                                'peak_fast': rows_of_interest.iloc[2,2]},
                               index = [0])
        peaks_jasp = pd.concat([peaks_jasp, new_row.loc[:]])
    peaks_jasp.to_csv(f'analysed/peaks_{mode}_{sensors}_{polarity}.csv')

else:
    print("The action-triggering threshold is not being tested.")
