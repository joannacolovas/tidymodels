#nested-resampling.R
#https://www.tidymodels.org/learn/work/nested-resampling/
#
#
#
#load packages
library(tidymodels)
library(furrr)
library(kernlab)
library(mlbench)
library(scales)

#simulate complex regression data from MARS publication--------------- 
sim_data <- function(n){
  tmp <- mlbench.friedman1(n, sd = 1)
  tmp <- cbind(tmp$x, tmp$y)
  tmp <- as.data.frame(tmp)
  names(tmp)[ncol(tmp)] <- "y"
  tmp
}

set.seed(9815)
train_dat <- sim_data(100)
large_dat <- sim_data(10^5)

#tune the modeling ---------------------------------------------------
# 5 repeats of 10 fold cross validation and 25 bootstraps
# 5 * 10 * 25 = 1250 models are fit per tuning parameter
results <- nested_cv(train_dat, 
                     outside = vfold_cv(repeats = 5), 
                     inside = bootstraps(times = 25))
results  

#split information for each resample is in split objects
#second fold, first repeat
results$splits[[2]]

#each inner_resamples element has own tibble with bootstrapping splits
#self contained = each one is a sample of a specific 90% of data
results$inner_resamples[[5]]

#define model creation and measurements-------------------------------

#radial SVM 
# 2 parameters = SVM cost, kernel sigma

# `object` will be an `rsplit` object from our `results` tibble
# `cost` is the tuning parameter
svm_rmse <- function(object, cost = 1) {
  y_col <- ncol(object$data)
  mod <- 
    svm_rbf(mode = "regression", cost = cost) %>% 
    set_engine("kernlab") %>% 
    fit(y ~ ., data = analysis(object))
  
  holdout_pred <- 
    predict(mod, assessment(object) %>% dplyr::select(-y)) %>% 
    bind_cols(assessment(object) %>% dplyr::select(y)) 
  rmse(holdout_pred, truth = y, estimate = .pred)$.estimate
}

#in some cases we want to parameterize the function over the tuning parameter
rmse_wrapper <- function(cost, object) {
  svm_rmse(object, cost)
}

#nested = model fit for each tuning parameter and each bootstrap split
#create a wrapper to do that

#`object` will be an `rsplit` object for the bootstrap samples
tune_over_cost <- function(object) {
  tibble(cost = 2 ^ seq(-2, 8, by = 1)) %>% 
    mutate(RMSE = map_dbl(cost, rmse_wrapper, object = object))
} 

# we will call tune_over_cost over the set of outer cross validation splits
# need 2nd wrapper for this 

#`object` will be an `rsplit` object in `results$inner_resamples`
summarize_tune_results <- function(object) {
  #return row-bound tibble that has the 25 bootstrap results
  map_df(object$splits, tune_over_cost) %>% 
    #for each value of tuning parameter, compute
    #avg RMSE which is the inner bootstrap estimate
    group_by(cost) %>% 
    summarize(mean_RMSE = mean(RMSE, na.rm = TRUE), 
              n = length(RMSE), 
              .groups = "drop")
}

#we can finally execute all the inner resampling loops 
tuning_results <- map(results$inner_resamples, summarize_tune_results)

#we could have also done these computations in parallel using furrr
plan(multisession)

future_tuning_results <- future_map(results$inner_resamples, summarize_tune_results)

# plot results ----------------------------------------------------------------
#the averaged results RMSE vs tuning params for each inner bootstrap
# Each gray line is a separate bootstrap resampling curve 
# created from a different 90% of the data. 
#The blue line is a LOESS smooth of all the results pooled together.

pooled_inner <- future_tuning_results %>% bind_rows

best_cost <- function(dat) {
  dat[which.min(dat$mean_RMSE), ]
}

p <- 
  ggplot(pooled_inner, aes(x = cost, y = mean_RMSE)) +
  scale_x_continuous(trans = 'log2') +
  xlab("SVM Cost") +
  ylab("Inner RMSE")

for (i in 1:length(future_tuning_results)) {
  p <- p + 
    geom_line(data = future_tuning_results[[i]], alpha = .2) +
    geom_point(data = best_cost(future_tuning_results[[i]]), 
               pch = 16 , alpha = 3/4)
}

p <- p + geom_smooth(data = pooled_inner, se = FALSE)
p

#estimate parameters w/ outer resampling/splits-----------------------------

#find/plot best parameter estimate from outer resampling
cost_vals <- 
  future_tuning_results %>% 
  map_df(best_cost) %>% 
  select(cost)

results <-
  bind_cols(results, cost_vals) %>% 
  mutate(cost = factor(cost, levels = paste(2 ^ seq(-2, 8, by = 1))))

ggplot(results, aes(x = cost)) +
  geom_bar() +
  xlab("SVM Cost") +
  scale_x_discrete(drop = FALSE)

#compute outer resampling results from 50 splits using tuned parameter values
results <- 
  results %>% 
  mutate(RMSE = map2_dbl(splits, cost, svm_rmse))

summary(results$RMSE)

#not nested RMSE estimate??---------------------------------------------------
not_nested <- 
  map(results$splits, tune_over_cost) %>% 
  bind_rows()

outer_summary <- not_nested %>% 
  group_by(cost) %>% 
  summarize(outer_RMSE = mean(RMSE), n = length(RMSE))

outer_summary

#plot it 
ggplot(outer_summary, aes(x = cost, y = outer_RMSE)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(trans = 'log2') +
  xlab("SVM Cost") + 
  ylab("RMSE")


#final modeling -----------------------------------------------------
finalModel <- 
  ksvm(y ~., data = train_dat, C = 2)
large_pred <- 
  predict(finalModel, large_dat[, -ncol(large_dat)])
sqrt(mean((large_dat$y - large_pred) ^ 2, na.rm = TRUE))
  