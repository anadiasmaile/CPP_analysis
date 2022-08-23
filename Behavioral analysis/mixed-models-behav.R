library(lme4) # run linear mixed models
library(afex) # control parameters and statistical testing
library(optimx) # control paramter: optimizer
library(ggplot2) # plotting
library(sjPlot) # plotting + diagnostics on lmer
library(plyr) # sorting data
library(DHARMa) # diagnostics on glmer

###############
## Read data ##
###############

# set working directory
setwd('~/Desktop/lorasick')
root = '.'

# root to files
files = list.files(root, pattern='rdm_beh.tsv', full.names=T, recursive=T)

# read out 68 files (22 participants x 3 sessions + 1 particpant x 2 sessions)
data <- do.call(rbind,lapply(files[c(1:27, 31:39, 43:72, 74:75, 121:123)],
                             read.csv, sep='\t')) 

# convert nan in NA
data[data == "nan"] <- NA

# delete trials with coherence = 0 since they have to contentual meaning for us
data <- data[!data$cur_coherence == 0.00,]

########################################
## Linear mixed effects Reaction Time ##
########################################

rt_full.mod <- afex::lmer(resp_time ~ 1 + cur_coherence +
                          (1 + cur_coherence|sub_id), data = data)
summary(rt_full.mod)
coef(rt_full.mod)

# descriptive statistics
ddply(data, "cur_coherence", summarise, mean = mean(resp_time, na.rm = TRUE),
      sd   = sd(resp_time, na.rm = TRUE)) 

################################################
## Checking the Assumptions for Reaction Time ##
################################################

# 1.residuals are (approximately) normally distributed

# extract residuals
res.mr <- residuals(rt_full.mod)

#plot
qqnorm(res.mr) 

# 2. fitted vs residual with smooth line added --> check linearity
plot(rt_full.mod, type = c("p", "smooth"))

# 3. scale-location plot --> check homoscedasticity
plot(rt_full.mod, sqrt(abs(resid(.)))~fitted(.), type=c("p","smooth"),
     col.line=1)

##################################
## Linear mixed effect Accuracy ##
##################################

acc_full.mod <- glmer(correct ~ 1 + cur_coherence + (1 + cur_coherence|sub_id),
                      data = data, family = binomial)
# Warning: singular fit = indicative of problems with estimation
# add control parameters to model: adjusting the optimizer
all_fit(acc_full.mod) 
acc_full.mod <- glmer(correct ~ 1 + cur_coherence + (1 + cur_coherence |sub_id),
                      data = data, family = binomial,
                      control = glmerControl(optimizer = "Nelder_Mead"))
summary(acc_full.mod)
coef(acc_full.mod)

# descriptive statistics
ddply(data, "cur_coherence", summarise, mean = mean(correct, na.rm = TRUE),
      sd   = sd(correct, na.rm = TRUE)) 

###########################################
## Checking the Assumptions for Accuracy ##
###########################################

simulationOutput <- simulateResiduals(fittedModel = acc_full.mod, plot = TRUE)

# extract residuals
res.mr <- residuals(acc_full.mod)

#plot
qqnorm(res.mr)

# KS, Dispersion and Outlier test n.s., homogenity given

# Dispersion
testDispersion(simulationOutput)

##########################
## Plotting the results ##
##########################

# Response Times
plot_data_resp <- ddply(data,.(sub_id,cur_coherence), summarize,
                        response = mean(resp_time, na.rm = TRUE))
ggplot(plot_data_resp, aes(x = cur_coherence, y = response, col = sub_id,
                           group = sub_id)) + geom_line ()+ geom_point()
plot_model(rt_full.mod, sort.est = TRUE, transform = NULL,
           show.intercept = TRUE, show.values = TRUE, value.offset = .3,
           title = "DV: Choice of left (0) versus right (1) option",
           colors = "bw", dot.size = 3, vline.color = "#9933FF", line.size = 1)
plot_model(rt_full.mod, type = "re")
plot_model(rt_full.mod, type = "pred", terms = "cur_coherence")

# Accuracy
plot_data_acc <- ddply(data,.(sub_id,cur_coherence), summarize,
                       response = mean(correct, na.rm = TRUE))
ggplot(plot_data_acc, aes(x = cur_coherence, y = response, col = sub_id,
                          group = sub_id)) + geom_line ()+ geom_point()
plot_model(acc_full.mod, sort.est = TRUE, transform = NULL,
           show.intercept = TRUE, show.values = TRUE, value.offset = .3,
           title = "DV: Choice of left (0) versus right (1) option",
           colors = "bw", dot.size = 3, vline.color = "#9933FF", line.size = 1)
plot_model(acc_full.mod, type = "re")
plot_model(acc_full.mod, type = "pred", terms = "cur_coherence") + geom_point()


