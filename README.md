# Neural markers of evidence accumulation during perceptual decision making in the MEG

The centro-parietal positivity (CPP) found in electroencephalography (EEG) recordings around the standard site CPz has been suggested to represent the human neural correlate of an evidence accumulating decision variable. Where and how the brain generates this signal remains unknown. Due to its higher spatial resolution and the opportunity for magnetic source imaging (MSI), this study tried to identify the CPP in magnetencephalography (MEG) recordings. For this purpose a perceptual decision task was implemented. The present repository consists of the custom scripts used for this analysis. First behavioral analyses were performed in R. Next MEG-recordings were analyzed by creating custom Python-scripts.

## Behavioral analysis
It was hypothesized, a greater strenght in sensory evidence would lead to faster response times and higher choice accuracy. Analysis were performed running mixed-effects modelling in the custom script "mixed-models-behav.R". This also entailed varifying the model's predictions.

## Neuromagnetic analysis
After preprocessing the MEG recordings, epochs needed to be converted to evoked objects by using the custom python script "turn_epochs_to_evokeds.py".
Since the measurement site of the neuromagnetic CPP equivalent was unknown, a permutation t-test was performed on the sensor data using the custom script "significance_testing.py". To varify if the found signal corresponded to the CPP, well-researched characteristics of an evidence accumulating signal were tested on the signal of interest. These analyses were performed using the custom script "hypotheses.py".

### Analyses settings
Within all three scripts, the "settings_cfg.json" allows modulating the analysis settings, e.g. sample size, without having to modify the original scripts. Further, "hypotheses_cfg.json" enables selecting which hypothesis to test.
