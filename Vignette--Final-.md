Implementing-Binary-Logistic-LASSO
================
Jia Yi ONG
04/DEC/2020

# Background

I implement the binary logistic LASSO using the coordinate descent and
Iteratively Reweighted Least Squares (IRLS) algorithms. In the following
section, I wrap my algorithm as a parsnip model and demonstrate how it
can be used in the tidymodels workflow.

# Introduction

We’ll fit a Penalized Logistic Classification model with Lasso
penalties. This model will train on a dataset about the career longevity
of NBA rookies, obtained from data.world’s Binary Classification
Exercise Dataset. The outcome variable is whether the rookie’s career
lasted for at least 5 years and the explanatory variables are various
traits and records.

We’ll require two packages.

``` r
library(tidyverse)
library(tidymodels)
```

The parsnip model is called logistic_lasso and it runs on the
fit_logistic_lasso computational engine. To access the model and its
engine, we need to source two scripts.

``` r
source("functions.R")
source("make_tidy.R")
```

Set seed.

``` r
set.seed(1003856614)
```

# Preparing the Data

Now we proceed to load and prepare the data. Load data with 1340 rows
and 21 columns.

``` r
load = read.csv("nba_logreg.csv", stringsAsFactors = F)
dim(load)
```

    ## [1] 1340   21

For demonstration, we’ll use a small subset of 500 rows selected
randomly.

``` r
load = sample_n(load, 500)
```

The dataset contains information on many potential predictors. To
simplify the demonstration, we’ll use only 3 main numeric explanatory
variables:

-   Outcome variable (TARGET): 1 for career lenght >= 5 years, 0
    otherwise.
-   Predictor 1 (GP): games played
-   Predictor 2 (PTS): points per game
-   Predictor 3 (X3P): three-pointers made

Subsetting variables

``` r
data = subset(load, select = c(TARGET_5Yrs, GP, PTS, X3P.Made))
names(data) = c("TARGET", "GP", "PTS", "X3P")
```

We’ll convert outcome into factors

``` r
data$TARGET = as.factor(data$TARGET)
```

Lastly, we will split the data into a training set that will train the
model and a test set to see the classification accuracy of the model.

``` r
split = initial_split(data)
train = training(split)
test = testing(split)
```

# Setting up the Model

As with all parsnip models, we require a recipe of steps, a model
specification, and a workflow to tie it all up.

We first speficy the formula for our model. We would like “TARGET” as
the outcome variable and the three variables as predictors. Next, our
recipe has a few steps to process the data that is fed to the model
through the workflow.

-   step_zv remomves variables that contain only a single value. We need
    variables with variation in the data.
-   step_normlize normalizes the variables by subtracting the mean and
    dividing by the standard deviation. We do not normalize the outcome
    variable.

``` r
rec = recipe(TARGET ~ ., data = train) %>% # formula
  step_zv(all_outcomes()) %>%
  step_normalize(all_numeric(), -TARGET)
```

Next, we specify the model and the engine that will be fitted to the
data. The parameter of the model, penalty, will be tuned so that the
best penalty-accuracy combination can be selected.

``` r
spec = logistic_lasso(penalty = tune()) %>% 
  set_engine("fit_logistic_lasso")
```

Lastly, we initialize a workflow and add the recipe and model
speficiation defined earlier.

``` r
workflow = workflow() %>% 
  add_recipe(rec) %>% 
  add_model(spec)
```

# Tuning the Penalty Parameter

The penalty parameter influences the complexity of the model. The higher
the penalty, the more likely the less important variables will be
excluded from the model, resulting in a more parsimonious model. To
balance model complexity and its predictive accuracy, we must tune the
penalty parameter.

We first create a grid of penalty values. Each penalty in the grid will
be tested, and the accuracy metric compared. The 15 penalties are evenly
spaced in log space.

``` r
penalty_grid = grid_regular(penalty(), levels = 15)
```

To get a good estimate of how well the model performs with each penalty,
we need to average the accuracy metric over many sub datasets. We split
the training data into 10 roughly even sub datasets.

``` r
folds = vfold_cv(train, v = 10)
```

Next, we tune the penalty parameter using the accuracy metric.

``` r
grid = tune_grid(workflow,
                 resamples = folds,
                 grid = penalty_grid,
                 metrics = metric_set(accuracy))
```

After tuning, we use this line of code to select the least complex model
(high penalty) from the models within the top 5% of the highest
accuracy.

``` r
best = grid %>% select_by_pct_loss(metric = "accuracy", limit = 5, desc(penalty))
best
```

    ## # A tibble: 1 x 9
    ##   penalty .metric  .estimator  mean     n std_err .config            .best .loss
    ##     <dbl> <chr>    <chr>      <dbl> <int>   <dbl> <chr>              <dbl> <dbl>
    ## 1  0.0373 accuracy binary     0.707    10  0.0274 Preprocessor1_Mod~ 0.728  2.91

We get the “best” penalty and its mean accuracy across the 10 sub
datasets. Updating this information into our workflow, we fit the
training set.

``` r
spec = logistic_lasso(penalty = best$penalty) %>% 
  set_engine("fit_logistic_lasso")

workflow = workflow() %>% 
  add_recipe(rec) %>% 
  add_model(spec) %>%
  fit(train)
```

We see that the logistic lasso may have dropped a few variables, the
variables with zero coefficients. This means that the variables are not
important in predicting an NBA rookie’s career length.

``` r
c(workflow$fit$fit$fit$intercept, workflow$fit$fit$fit$beta)
```

    ## Intercept        GP       PTS       X3P 
    ## 0.7166133 0.5340289 0.3639221 0.0000000

Let’s see how our model perform in classifying points in the test set
using a Confusion Matrix.

``` r
predict(workflow, new_data = test) %>% bind_cols(test %>% select(TARGET)) %>%
  conf_mat(truth = TARGET, estimate = .pred_class)
```

    ##           Truth
    ## Prediction  0  1
    ##          0 12  6
    ##          1 44 63

The model’s prediction rates can be computed:

-   True negative rate: 22/(22+24) = 0.478
-   True positive rate: 70/(70+9) = 0.886
-   False negative rate: 9/(70+9) = 0.114
-   False positive rate: 24/(22+24) = 0.522

Although not perfect, the true positive rate is quite high and the model
predicted the NBA rookie’s success correctly 88.6 percent of the time,
using only information about the number of games played.

# Comments

-   To improve predictability, more predictors should be used instead of
    the mere 3 predictors in this demonstration.

-   Choosing the right balance between model complexity and prediction
    accuracy is still an art. You can modify the range of accuracies
    from which to select the best penalty parameter from, depending on
    your requirements.

-   Logistic Lasso can “sieve” out unessential variables, so you may
    attempt many predictors and the model will serve as a variable
    selector.

# References

data.world. Binary Classification Exercise Dataset. Retrieved from
<https://data.world/exercises/logistic-regression-exercise-1/access>
