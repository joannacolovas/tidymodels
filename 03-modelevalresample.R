#03-modelevalresample
#https://www.tidymodels.org/start/resampling/
#
#
#load packages
library(tidymodels) #main
library(modeldata) #cells data
library(ranger) #use? 

#load data into r - cell imaging data
data(cells, package = "modeldata")
cells

#count the cells by segmentation type 
cells %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

#set seed and split the dataset into train/test
set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)
cell_train <- training(cell_split)
cell_test <- testing(cell_split)

nrow(cell_train)
nrow(cell_train)/nrow(cells)

#training set proportions by class
cell_train %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

#test set proportions by class
cell_test %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

#define random forest model
rf_mod <- 
  rand_forest(trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

set.seed(234)

#fit model
rf_fit <- 
  rf_mod %>% 
  fit(class ~ ., data = cell_train)
rf_fit

#predict on training data
rf_training_pred <- 
  predict(rf_fit, cell_train) %>% 
  bind_cols(predict(rf_fit, cell_train, type = "prob")) %>% 
  #add the true outcome data back in 
  bind_cols(cell_train %>% select(class))

#yardstick for metrics on training set (don't do this it's bad)
rf_training_pred %>% 
  roc_auc(truth = class, .pred_PS)

rf_training_pred %>% 
  accuracy(truth =  class, .pred_class)

#predict on test data
rf_testing_pred <- 
  predict(rf_fit, cell_test) %>% 
  bind_cols(predict(rf_fit, cell_test, type = "prob")) %>% 
  #add the true outcome data back in 
  bind_cols(cell_test %>% select(class)) 

#yardstick for metrics on model performance on test set
rf_testing_pred %>% 
  roc_auc(truth = class, .pred_PS)

rf_testing_pred %>% 
  accuracy(truth = class, .pred_class)

#generate cross validation folds
set.seed(345)

folds <- vfold_cv(cell_train, v = 10)
folds

#create workflow and apply it on the resampled folds
rf_wf <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_formula(class ~ .)

set.seed(456)
rf_fit_rs <- 
  rf_wf %>% 
  fit_resamples(folds)
rf_fit_rs

collect_metrics(rf_fit_rs)



