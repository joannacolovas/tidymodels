#02-recipes.R
#https://www.tidymodels.org/start/recipes/
#
#
#load libraries
library(tidymodels) #basic pkg
library(nycflights13) #flight data
library(skimr) #variable summaries


set.seed(123)

flight_data <- 
  flights %>% 
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = lubridate::as_date(time_hour)
  ) %>% 
  # Include the weather data
  inner_join(weather, by = c("origin", "time_hour")) %>% 
  # Only retain the specific columns we will use
  select(dep_time, flight, origin, dest, air_time, distance, 
         carrier, date, arr_delay, time_hour) %>% 
  # Exclude missing data
  na.omit() %>% 
  # For creating models, it is better to have qualitative columns
  # encoded as factors (instead of character strings)
  mutate_if(is.character, as.factor)


#see how many flights arrived >30 mins late
flight_data %>% 
  count(arr_delay) %>% 
  mutate(prop = n/sum(n))

#data splitting into training and test set

#fix random numbers by setting the seed to reproduce ananlysis 
set.seed(222)

#put 3/4 of data in training set
data_split <- initial_split(flight_data, prop = 3/4)

#create dfs for two sets
train_data <- training(data_split)
test_data <- testing(data_split)

#create new recipe for simple logistic regression model 
flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  #features for the date column
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US"), 
               keep_original_cols = FALSE) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())

summary(flights_rec)


#build a model for the recipe
lr_mod <- 
  logistic_reg() %>% 
  set_engine("glm")

#set up workflow with recipe and model
flights_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(flights_rec)
flights_wflow

#prepare recipe and train model
flights_fit <- 
  flights_wflow %>% 
  fit(data = train_data)

#look at the trained model
flights_fit %>% 
  extract_fit_parsnip() %>% 
  tidy()

#use model to predict test set
predict(flights_fit, test_data)

#add probabilities to prediciton
flights_aug <- 
  augment(flights_fit, test_data)

#data look like
flights_aug %>% 
  select(arr_delay, time_hour, flight, .pred_class, .pred_on_time)

#look at the fit of the model
#roc curve graph 
flights_aug %>% 
  roc_curve(truth = arr_delay, .pred_late) %>% 
  autoplot()

#roc metrics
flights_aug %>% 
  roc_auc(truth = arr_delay, .pred_late)
