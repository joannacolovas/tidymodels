#05-casestudy
#https://www.tidymodels.org/start/case-study/
#
#
#library statements
library(tidymodels)
library(readr)
library(vip)

#read in hotel bookings data set 
hotels <- 
  read_csv("https://tidymodels.org/start/case-study/hotels.csv") %>% 
  mutate(across(where(is.character), as.factor))

dim(hotels)
glimpse(hotels)

#children outcome variable factor with 2 levels
hotels %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))

#split dataset using stratified random sample
set.seed(123)
split <- initial_split(hotels, strata = children)

hotel_other <- training(split)
hotel_test <- testing(split)

#training and test set proportions by children, make sure they are equivalent
hotel_other %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))

hotel_test %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))


#use validation_split() to create validation and training sets from other
#validation_split was deprecated, use initial_validation_split instead of initial_split
set.seed(234)
val_set <- validation_split(hotel_other, 
                            strata = children, 
                            prop = 0.80)
val_set

#build penalized logistic regression model with glmnet, mixture = 1 simplifies model 
lr_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

#create recipe for data preprocessing
holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter",
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

lr_recipe <- 
  recipe(children ~ ., data = hotel_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date, holidays = holidays) %>% 
  step_rm(arrival_date) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())
lr_recipe

#create workflow for model and recipe
lr_workflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_recipe)

#create grid for tuning 
lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))
lr_reg_grid %>% top_n(-5) #lowest penalty values
lr_reg_grid %>% top_n(5) #highest penalty values

#tune and train model
lr_res <- 
  lr_workflow %>% 
  tune_grid(val_set, 
            grid = lr_reg_grid, 
            control = control_grid(save_pred = TRUE), 
            metrics = metric_set(roc_auc))

#visualize metrics/penalties
lr_plot <- 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

lr_plot 

#see top models
top_models <- 
  lr_res %>% 
  show_best(metric = "roc_auc", n = 15) %>% 
  arrange(penalty)
top_models


#select model to use, i don't have the same one as the tutorial but close enough
lr_best <- 
  lr_res %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  slice(12)
lr_best

#visualize model
lr_auc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Logistic Regression")

autoplot(lr_auc)


#second model - random forest

#find out how many cores i have
cores <- parallel::detectCores()
cores

#set up rf model
rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("classification")

#create recipe and wf
rf_recipe <- 
  recipe(children ~ ., data = hotel_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date) %>% 
  step_rm(arrival_date) 

rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)

#train and tune model 
rf_mod

#show what will be tuned
extract_parameter_set_dials(rf_mod)

set.seed(345)
rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set, 
            grid = 25, 
            control = control_grid(save_pred = TRUE), 
            metrics = metric_set(roc_auc))

#get top models
rf_res %>% 
  show_best(metric = "roc_auc")

autoplot(rf_res)

#select best model 
rf_best <- 
  rf_res %>% 
  select_best(metric = "roc_auc")
rf_best

#calculate and plot 
rf_res %>% 
  collect_predictions()

rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Random Forest")

#plot lr vs rf
bind_rows(rf_auc, lr_auc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() + 
  scale_color_viridis_d(option = "plasma", end = .6)

# the last model
last_rf_mod <- 
  rand_forest(mtry = 8, min_n = 7, trees = 1000) %>% 
  set_engine("ranger", num.threads = cores, importance = "impurity") %>% 
  set_mode("classification")

#last wf
last_rf_workflow <- 
  rf_workflow %>% 
  update_model(last_rf_mod)

#last fit
set.seed(345)
last_rf_fit <- 
  last_rf_workflow %>% 
  last_fit(split)

last_rf_fit 

#collect metrics and extract fit
last_rf_fit %>% 
  collect_metrics()

last_rf_fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 20)

#final roc curve
last_rf_fit %>% 
  collect_predictions() %>% 
  roc_curve(children, .pred_children) %>% 
  autoplot()
