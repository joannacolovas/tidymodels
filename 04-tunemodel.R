#04-tunemodel
#https://www.tidymodels.org/start/tuning/
#
#library statements
library(tidymodels)
library(rpart)
library(rpart.plot)
library(vip)

#cell segmentation data
data(cells, package = "modeldata")
cells

#set seed and set up test/train split
set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)
cell_train <- training(cell_split)
cell_test <- testing(cell_split)

#create a descision tree model with rpart engine 
tune_spec <- 
  decision_tree(
    cost_complexity = tune(), 
    tree_depth = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

tune_spec

#regular grid of values for each hyperparameter
#5x5 = 25 options total 
tree_grid <- grid_regular(cost_complexity(), 
                          tree_depth(), 
                          levels = 5)
tree_grid

#create cross validation folds for tuning 
set.seed(234)
cell_folds <- vfold_cv(cell_train)

#tune the model with a workflow 
set.seed(345)

tree_wf <- workflow() %>% 
  add_model(tune_spec) %>% 
  add_formula(class ~ .)

tree_res <- 
  tree_wf %>% 
  tune_grid(
    resamples = cell_folds, 
    grid = tree_grid
  )

tree_res

#collect metrics and visualize
tree_res %>% 
  collect_metrics()

#plot
tree_res %>%
  collect_metrics() %>%
  mutate(tree_depth = factor(tree_depth)) %>%
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line(linewidth = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

#show best values
tree_res %>% 
  show_best(metric = "accuracy")

best_tree <- tree_res %>% 
  select_best(metric = "accuracy")

best_tree

#update/finalize workflow with best tree parameters
final_wf <- 
  tree_wf %>% 
  finalize_workflow(best_tree)

final_wf

#fit model to training data and test on testing data to see fit 
final_fit <- 
  final_wf %>% 
  last_fit(cell_split)

final_fit %>% 
  collect_metrics()

final_fit %>% 
  collect_predictions() %>% 
  roc_curve(class, .pred_PS) %>% 
  autoplot()

#extract the final fit workflow
final_tree <- extract_workflow(final_fit)
final_tree

#visualize decision tree
final_tree %>% 
  extract_fit_engine() %>% 
  rpart.plot(roundint = FALSE)

#estimate variable importance 
final_tree %>% 
  extract_fit_parsnip() %>% 
  vip()

#can you tune other hyperparameters?
args(decision_tree)
#yes, min_n