#performs the LOESS smoothing and creates calibration curves and integrated calibration indexes for the 
#predictions of the algorithms over the bootstrap
library(ggplot2)
library(tidyverse)

#functions that do the legwork of the analysis
ICE <- function(bootstrapped_predictions) {
  #this function reads the prediction dataframe and returns ICE for each bootstrap
  vec <- rep(0, ncol(bootstrapped_predictions)%/%2)
  for(n in 1:(ncol(bootstrapped_predictions)%/%2)) {
    temp_df <- bootstrapped_predictions[,(2*n-1):(2*n)]
    colnames(temp_df) <- c('prob', 'deceased')
    temp_df <- temp_df[order(temp_df$prob),]
    loessMod <- loess(deceased ~ prob, data = temp_df, span = 0.75, 
                      control = loess.control(surface = "direct"))
    smoothed <- predict(loessMod, temp_df$prob)
    temp_list <- abs(smoothed - temp_df$prob)
    int_cal_index <- mean(temp_list)
    vec[n] = int_cal_index
  }
  return(data.frame('integrated_calibration_index' = vec))
}

cali_curve <- function(bootstrapped_predictions) {
  #this function reads the predictions dataframe and returns coordinates of the loess
  #smother for plotting
  df <- data.frame('predicted_probability' = seq(0,1, 0.001))
  for(n in 1:(ncol(bootstrapped_predictions)%/%2)) {
    temp_df <- bootstrapped_predictions[,(2*n-1):(2*n)]
    colnames(temp_df) <- c('prob', 'deceased')
    temp_df <- temp_df[order(temp_df$prob),]
    loessMod <- loess(deceased ~ prob, data = temp_df, span = 0.75, 
                      control = loess.control(surface = "direct"))
    smoothed <- predict(loessMod, data.frame(prob = seq(0, 1, 0.001)))
    observed_n <- paste('observed_', as.character(n), sep = "")
    print(observed_n)
    df[observed_n] <- smoothed
  }
  return(df)
}

#initial analysis
initial_lr <- read.csv(file = 'predictions/initial_results/lr_predictions.csv')
initial_rf <- read.csv(file = 'predictions/initial_results/rf_predictions.csv')
initial_sv <- read.csv(file = 'predictions/initial_results/sv_predictions.csv')
initial_nn <- read.csv(file = 'predictions/initial_results/nn_predictions.csv')
initial_ec <- read.csv(file = 'predictions/initial_results/ec_predictions.csv')

initial_lr_ice <- ICE(initial_lr)
initial_rf_ice <- ICE(initial_rf)
initial_sv_ice <- ICE(initial_sv)
initial_nn_ice <- ICE(initial_nn)
initial_ec_ice <- ICE(initial_ec)

initial_lr_cali <- cali_curve(initial_lr)
initial_rf_cali <- cali_curve(initial_rf)
initial_sv_cali <- cali_curve(initial_sv)
initial_nn_cali <- cali_curve(initial_nn)
initial_ec_cali <- cali_curve(initial_ec)

write.csv(initial_lr_ice, 'predictions/initial_results/lr_ice.csv', row.names = FALSE)
write.csv(initial_rf_ice, 'predictions/initial_results/rf_ice.csv', row.names = FALSE)
write.csv(initial_sv_ice, 'predictions/initial_results/sv_ice.csv', row.names = FALSE)
write.csv(initial_nn_ice, 'predictions/initial_results/nn_ice.csv', row.names = FALSE)
write.csv(initial_ec_ice, 'predictions/initial_results/ec_ice.csv', row.names = FALSE)

write.csv(initial_lr_cali, 'predictions/initial_results/lr_cali.csv', row.names = FALSE)
write.csv(initial_rf_cali, 'predictions/initial_results/rf_cali.csv', row.names = FALSE)
write.csv(initial_sv_cali, 'predictions/initial_results/sv_cali.csv', row.names = FALSE)
write.csv(initial_nn_cali, 'predictions/initial_results/nn_cali.csv', row.names = FALSE)
write.csv(initial_ec_cali, 'predictions/initial_results/ec_cali.csv', row.names = FALSE)

#sample size analysis
sample_lr_100 <- read.csv(file = 'predictions/datapoint_results_100/lr_predictions.csv')
sample_lr_200 <- read.csv(file = 'predictions/datapoint_results_200/lr_predictions.csv')
sample_lr_400 <- read.csv(file = 'predictions/datapoint_results_400/lr_predictions.csv')
sample_lr_800 <- read.csv(file = 'predictions/datapoint_results_800/lr_predictions.csv')
sample_lr_1600 <- read.csv(file = 'predictions/datapoint_results_1600/lr_predictions.csv')
sample_lr_3200 <- read.csv(file = 'predictions/datapoint_results_3200/lr_predictions.csv')

sample_rf_100 <- read.csv(file = 'predictions/datapoint_results_100/rf_predictions.csv')
sample_rf_200 <- read.csv(file = 'predictions/datapoint_results_200/rf_predictions.csv')
sample_rf_400 <- read.csv(file = 'predictions/datapoint_results_400/rf_predictions.csv')
sample_rf_800 <- read.csv(file = 'predictions/datapoint_results_800/rf_predictions.csv')
sample_rf_1600 <- read.csv(file = 'predictions/datapoint_results_1600/rf_predictions.csv')
sample_rf_3200 <- read.csv(file = 'predictions/datapoint_results_3200/rf_predictions.csv')

sample_sv_100 <- read.csv(file = 'predictions/datapoint_results_100/sv_predictions.csv')
sample_sv_200 <- read.csv(file = 'predictions/datapoint_results_200/sv_predictions.csv')
sample_sv_400 <- read.csv(file = 'predictions/datapoint_results_400/sv_predictions.csv')
sample_sv_800 <- read.csv(file = 'predictions/datapoint_results_800/sv_predictions.csv')
sample_sv_1600 <- read.csv(file = 'predictions/datapoint_results_1600/sv_predictions.csv')
sample_sv_3200 <- read.csv(file = 'predictions/datapoint_results_3200/sv_predictions.csv')

sample_nn_100 <- read.csv(file = 'predictions/datapoint_results_100/nn_predictions.csv')
sample_nn_200 <- read.csv(file = 'predictions/datapoint_results_200/nn_predictions.csv')
sample_nn_400 <- read.csv(file = 'predictions/datapoint_results_400/nn_predictions.csv')
sample_nn_800 <- read.csv(file = 'predictions/datapoint_results_800/nn_predictions.csv')
sample_nn_1600 <- read.csv(file = 'predictions/datapoint_results_1600/nn_predictions.csv')
sample_nn_3200 <- read.csv(file = 'predictions/datapoint_results_3200/nn_predictions.csv')

sample_ec_100 <- read.csv(file = 'predictions/datapoint_results_100/ec_predictions.csv')
sample_ec_200 <- read.csv(file = 'predictions/datapoint_results_200/ec_predictions.csv')
sample_ec_400 <- read.csv(file = 'predictions/datapoint_results_400/ec_predictions.csv')
sample_ec_800 <- read.csv(file = 'predictions/datapoint_results_800/ec_predictions.csv')
sample_ec_1600 <- read.csv(file = 'predictions/datapoint_results_1600/ec_predictions.csv')
sample_ec_3200 <- read.csv(file = 'predictions/datapoint_results_3200/ec_predictions.csv')

lr_ice_s100 <- ICE(sample_lr_100)
rf_ice_s100 <- ICE(sample_rf_100)
sv_ice_s100 <- ICE(sample_sv_100)
nn_ice_s100 <- ICE(sample_nn_100)
ec_ice_s100 <- ICE(sample_ec_100)

lr_ice_s200 <- ICE(sample_lr_200)
rf_ice_s200 <- ICE(sample_rf_200)
sv_ice_s200 <- ICE(sample_sv_200)
nn_ice_s200 <- ICE(sample_nn_200)
ec_ice_s200 <- ICE(sample_ec_200)

lr_ice_s400 <- ICE(sample_lr_400)
rf_ice_s400 <- ICE(sample_rf_400)
sv_ice_s400 <- ICE(sample_sv_400)
nn_ice_s400 <- ICE(sample_nn_400)
ec_ice_s400 <- ICE(sample_ec_400)

lr_ice_s800 <- ICE(sample_lr_800)
rf_ice_s800 <- ICE(sample_rf_800)
sv_ice_s800 <- ICE(sample_sv_800)
nn_ice_s800 <- ICE(sample_nn_800)
ec_ice_s800 <- ICE(sample_ec_800)

lr_ice_s1600 <- ICE(sample_lr_1600)
rf_ice_s1600 <- ICE(sample_rf_1600)
sv_ice_s1600 <- ICE(sample_sv_1600)
nn_ice_s1600 <- ICE(sample_nn_1600)
ec_ice_s1600 <- ICE(sample_ec_1600)

lr_ice_s3200 <- ICE(sample_lr_3200)
rf_ice_s3200 <- ICE(sample_rf_3200)
sv_ice_s3200 <- ICE(sample_sv_3200)
nn_ice_s3200 <- ICE(sample_nn_3200)
ec_ice_s3200 <- ICE(sample_ec_3200)

write.csv(lr_ice_s100, 'predictions/datapoint_results_100/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_s100, 'predictions/datapoint_results_100/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_s100, 'predictions/datapoint_results_100/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_s100, 'predictions/datapoint_results_100/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_s100, 'predictions/datapoint_results_100/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_s200, 'predictions/datapoint_results_200/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_s200, 'predictions/datapoint_results_200/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_s200, 'predictions/datapoint_results_200/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_s200, 'predictions/datapoint_results_200/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_s200, 'predictions/datapoint_results_200/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_s400, 'predictions/datapoint_results_400/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_s400, 'predictions/datapoint_results_400/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_s400, 'predictions/datapoint_results_400/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_s400, 'predictions/datapoint_results_400/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_s400, 'predictions/datapoint_results_400/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_s800, 'predictions/datapoint_results_800/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_s800, 'predictions/datapoint_results_800/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_s800, 'predictions/datapoint_results_800/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_s800, 'predictions/datapoint_results_800/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_s800, 'predictions/datapoint_results_800/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_s1600, 'predictions/datapoint_results_1600/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_s1600, 'predictions/datapoint_results_1600/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_s1600, 'predictions/datapoint_results_1600/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_s1600, 'predictions/datapoint_results_1600/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_s1600, 'predictions/datapoint_results_1600/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_s3200, 'predictions/datapoint_results_3200/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_s3200, 'predictions/datapoint_results_3200/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_s3200, 'predictions/datapoint_results_3200/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_s3200, 'predictions/datapoint_results_3200/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_s3200, 'predictions/datapoint_results_3200/ec_ice.csv', row.names = FALSE)

#feature analysis
f_lr_10 <- read.csv(file = 'predictions/feature_results_10/lr_predictions.csv')
f_lr_25 <- read.csv(file = 'predictions/feature_results_25/lr_predictions.csv')
f_lr_50 <- read.csv(file = 'predictions/feature_results_50/lr_predictions.csv')
f_lr_100 <- read.csv(file = 'predictions/feature_results_100/lr_predictions.csv')
f_lr_250 <- read.csv(file = 'predictions/feature_results_250/lr_predictions.csv')
f_lr_500 <- read.csv(file = 'predictions/feature_results_500/lr_predictions.csv')

f_rf_10 <- read.csv(file = 'predictions/feature_results_10/rf_predictions.csv')
f_rf_25 <- read.csv(file = 'predictions/feature_results_25/rf_predictions.csv')
f_rf_50 <- read.csv(file = 'predictions/feature_results_50/rf_predictions.csv')
f_rf_100 <- read.csv(file = 'predictions/feature_results_100/rf_predictions.csv')
f_rf_250 <- read.csv(file = 'predictions/feature_results_250/rf_predictions.csv')
f_rf_500 <- read.csv(file = 'predictions/feature_results_500/rf_predictions.csv')

f_sv_10 <- read.csv(file = 'predictions/feature_results_10/sv_predictions.csv')
f_sv_25 <- read.csv(file = 'predictions/feature_results_25/sv_predictions.csv')
f_sv_50 <- read.csv(file = 'predictions/feature_results_50/sv_predictions.csv')
f_sv_100 <- read.csv(file = 'predictions/feature_results_100/sv_predictions.csv')
f_sv_250 <- read.csv(file = 'predictions/feature_results_250/sv_predictions.csv')
f_sv_500 <- read.csv(file = 'predictions/feature_results_500/sv_predictions.csv')

f_nn_10 <- read.csv(file = 'predictions/feature_results_10/nn_predictions.csv')
f_nn_25 <- read.csv(file = 'predictions/feature_results_25/nn_predictions.csv')
f_nn_50 <- read.csv(file = 'predictions/feature_results_50/nn_predictions.csv')
f_nn_100 <- read.csv(file = 'predictions/feature_results_100/nn_predictions.csv')
f_nn_250 <- read.csv(file = 'predictions/feature_results_250/nn_predictions.csv')
f_nn_500 <- read.csv(file = 'predictions/feature_results_500/nn_predictions.csv')

f_ec_10 <- read.csv(file = 'predictions/feature_results_10/ec_predictions.csv')
f_ec_25 <- read.csv(file = 'predictions/feature_results_25/ec_predictions.csv')
f_ec_50 <- read.csv(file = 'predictions/feature_results_50/ec_predictions.csv')
f_ec_100 <- read.csv(file = 'predictions/feature_results_100/ec_predictions.csv')
f_ec_250 <- read.csv(file = 'predictions/feature_results_250/ec_predictions.csv')
f_ec_500 <- read.csv(file = 'predictions/feature_results_500/ec_predictions.csv')

lr_ice_f10 <- ICE(f_lr_10)
rf_ice_f10 <- ICE(f_rf_10)
sv_ice_f10 <- ICE(f_sv_10)
nn_ice_f10 <- ICE(f_nn_10)
ec_ice_f10 <- ICE(f_ec_10)

lr_ice_f25 <- ICE(f_lr_25)
rf_ice_f25 <- ICE(f_rf_25)
sv_ice_f25 <- ICE(f_sv_25)
nn_ice_f25 <- ICE(f_nn_25)
ec_ice_f25 <- ICE(f_ec_25)

lr_ice_f50 <- ICE(f_lr_50)
rf_ice_f50 <- ICE(f_rf_50)
sv_ice_f50 <- ICE(f_sv_50)
nn_ice_f50 <- ICE(f_nn_50)
ec_ice_f50 <- ICE(f_ec_50)

lr_ice_f100 <- ICE(f_lr_100)
rf_ice_f100 <- ICE(f_rf_100)
sv_ice_f100 <- ICE(f_sv_100)
nn_ice_f100 <- ICE(f_nn_100)
ec_ice_f100 <- ICE(f_ec_100)

lr_ice_f250 <- ICE(f_lr_250)
rf_ice_f250 <- ICE(f_rf_250)
sv_ice_f250 <- ICE(f_sv_250)
nn_ice_f250 <- ICE(f_nn_250)
ec_ice_f250 <- ICE(f_ec_250)

lr_ice_f500 <- ICE(f_lr_500)
rf_ice_f500 <- ICE(f_rf_500)
sv_ice_f500 <- ICE(f_sv_500)
nn_ice_f500 <- ICE(f_nn_500)
ec_ice_f500 <- ICE(f_ec_500)

write.csv(lr_ice_f10, 'predictions/feature_results_10/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_f10, 'predictions/feature_results_10/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_f10, 'predictions/feature_results_10/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_f10, 'predictions/feature_results_10/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_f10, 'predictions/feature_results_10/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_f25, 'predictions/feature_results_25/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_f25, 'predictions/feature_results_25/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_f25, 'predictions/feature_results_25/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_f25, 'predictions/feature_results_25/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_f25, 'predictions/feature_results_25/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_f50, 'predictions/feature_results_50/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_f50, 'predictions/feature_results_50/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_f50, 'predictions/feature_results_50/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_f50, 'predictions/feature_results_50/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_f50, 'predictions/feature_results_50/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_f100, 'predictions/feature_results_100/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_f100, 'predictions/feature_results_100/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_f100, 'predictions/feature_results_100/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_f100, 'predictions/feature_results_100/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_f100, 'predictions/feature_results_100/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_f250, 'predictions/feature_results_250/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_f250, 'predictions/feature_results_250/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_f250, 'predictions/feature_results_250/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_f250, 'predictions/feature_results_250/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_f250, 'predictions/feature_results_250/ec_ice.csv', row.names = FALSE)

write.csv(lr_ice_f500, 'predictions/feature_results_500/lr_ice.csv', row.names = FALSE)
write.csv(rf_ice_f500, 'predictions/feature_results_500/rf_ice.csv', row.names = FALSE)
write.csv(sv_ice_f500, 'predictions/feature_results_500/sv_ice.csv', row.names = FALSE)
write.csv(nn_ice_f500, 'predictions/feature_results_500/nn_ice.csv', row.names = FALSE)
write.csv(ec_ice_f500, 'predictions/feature_results_500/ec_ice.csv', row.names = FALSE)

#pca analysis
pca_lr <- read.csv(file = 'predictions/pca_results/lr_predictions.csv')
pca_rf <- read.csv(file = 'predictions/pca_results/rf_predictions.csv')
pca_sv <- read.csv(file = 'predictions/pca_results/sv_predictions.csv')
pca_nn <- read.csv(file = 'predictions/pca_results/nn_predictions.csv')
pca_ec <- read.csv(file = 'predictions/pca_results/ec_predictions.csv')

pca_lr_ice <- ICE(pca_lr)
pca_rf_ice <- ICE(pca_rf)
pca_sv_ice <- ICE(pca_sv)
pca_nn_ice <- ICE(pca_nn)
pca_ec_ice <- ICE(pca_ec)

write.csv(pca_lr_ice, 'predictions/pca_results/lr_ice.csv', row.names = FALSE)
write.csv(pca_rf_ice, 'predictions/pca_results/rf_ice.csv', row.names = FALSE)
write.csv(pca_sv_ice, 'predictions/pca_results/sv_ice.csv', row.names = FALSE)
write.csv(pca_nn_ice, 'predictions/pca_results/nn_ice.csv', row.names = FALSE)
write.csv(pca_ec_ice, 'predictions/pca_results/ec_ice.csv', row.names = FALSE)