#01-buildmodel
#https://www.tidymodels.org/start/models/
#
#load libraries
library(tidymodels) #main pkg
library(readr) #importing data
library(broom.mixed) #converting bayesian models to tidy tibble
library(dotwhisker) #visualize regression results

#we don't actually want to use the keras engine, it's just an option
#library(keras)
#library(tensorflow)

#read sea urchin data for demo
urchins <- read_csv("https://tidymodels.org/start/models/urchins.csv") %>% 
  #change colnames
  setNames(c("food_regime", "initial_volume", "width")) %>% 
  #convert one col to factor for modeling
  mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))
  
#plot data to have a good idea of what it looks like
ggplot(urchins, 
       aes(x = initial_volume, 
           y = width, 
           group = food_regime, 
           col = food_regime)) +
  geom_point() + 
  # geom smooth() using formula y ~ x to make general trendlines
  geom_smooth(method = lm, se = FALSE) +
  scale_color_viridis_d(option = "plasma", end = 0.7)


#make the basic model that we want to use, 
#populates model with engine, we don't actually want to use keras
lm_mod <- linear_reg() %>% 
  set_engine("lm")

#fits data to model using specified model with "fit()"
lm_fit <- 
  lm_mod %>% 
  fit(formula = width ~ initial_volume * food_regime, data = urchins)

#print model and tidied version of the model
lm_fit
tidy(lm_fit)

#visualize the model
tidy(lm_fit) %>% 
  dwplot(dot_args = list(size = 2, color = "black"),
         whisker_args = list(color = "black"),
         vline = geom_vline(xintercept = 0, colour = "grey50", linetype = 2))

#plot mean body size for 20ml volume urchins at t= 0
new_points <- expand.grid(initial_volume = 20, 
                          food_regime = c("Initial", "Low", "High"))
new_points

#use predict() to find the mean values at 20ml
mean_pred <- predict(lm_fit, new_data = new_points)
mean_pred

#combine new and original data in a useable format 
#including confidence intervals
conf_int_pred <- predict(lm_fit, 
                         new_data = new_points, 
                         type = "conf_int")
conf_int_pred

#combine data 
plot_data <- 
  new_points %>% 
  bind_cols(mean_pred) %>% 
  bind_cols(conf_int_pred)

#and plot
ggplot(plot_data, aes(x = food_regime)) +
  geom_point(aes(y = .pred)) +
  geom_errorbar(aes(ymin = .pred_lower, 
                    ymax = .pred_upper), 
                width = 0.2) +
  labs(y = "urchin size")

#bayesian statistics approach

#set the prior distribution 
prior_dist <- rstanarm::student_t(df=1)
set.seed(123)

#make the parsnip model
bayes_model <- 
  linear_reg() %>% 
  set_engine("stan", 
             prior_intercept = prior_dist, 
             prior = prior_dist)

#train the model which does not work as of 20240402
bayes_fit <- 
  bayes_model %>% 
  fit(formula = width ~ initial_volume * food_regime, data = urchins)

print(bayes_fit, digits = 5)
