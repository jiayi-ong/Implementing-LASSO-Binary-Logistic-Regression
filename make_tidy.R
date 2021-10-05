
logistic_lasso = function(mode = "classification", penalty) {
  
  ## logistic_lasso is a parsnip logistic lasso classification model
  ## which runs on the fit_logistic_lasso computational engine.
  ## 
  ## Input:
  ## - penalty: a number indicating the lasso penalization.
  ##
  ## Output:
  ## - A Model Specification object.
  ##
  ## Example:
  ##   library(tidyverse); library(tidymodels)
  ##
  ##   split = initial_split(data, strata = c("sex"))
  ##   train = training(split)
  ##
  ##   rec = recipe(y ~ . , data = train)
  ##   spec = logistic_lasso(penalty = 0.2) %>% 
  ##      set_engine("fit_logistic_lasso")
  ##   workflow = workflow() %>% 
  ##      add_recipe(rec) %>% 
  ##      add_model(spec) %>%
  ##      fit(train)
  
  args = list(penalty = rlang::enquo(penalty))
  
  new_model_spec("logistic_lasso",
                 args = args,
                 mode = mode,
                 eng_args = NULL,
                 method = NULL,
                 engine = NULL)
}

# __________Start: Registering the model__________
set_new_model("logistic_lasso")

set_model_mode(model = "logistic_lasso", 
               mode = "classification")

set_model_engine("logistic_lasso",
                 mode = "classification",
                 eng = "fit_logistic_lasso")

set_dependency("logistic_lasso", 
               eng = "fit_logistic_lasso", 
               pkg = "base")
# __________End: Registering the model__________

set_model_arg(
  model = "logistic_lasso",
  eng = "fit_logistic_lasso",
  parsnip = "penalty",
  original = "lambda",
  func = list(pkg = "dials", fun = "penalty"),
  has_submodel = FALSE
)

set_encoding(
  model = "logistic_lasso",
  eng = "fit_logistic_lasso",
  mode = "classification",
  options = list(
    predictor_indicators = "traditional",
    compute_intercept = TRUE,
    remove_intercept = TRUE,
    allow_sparse_x = FALSE
  )
)

set_fit(
  model = "logistic_lasso",
  eng = "fit_logistic_lasso",
  mode = "classification",
  value = list(
    interface = "matrix",
    protect = c("x", "y"),
    func = c(fun = "fit_logistic_lasso"),
    defaults = list()
  ))

set_pred(
  model = "logistic_lasso",
  eng = "fit_logistic_lasso",
  mode = "classification",
  type = "class",
  value = list(
    pre = NULL,
    post = NULL,
    func = c(fun = "predict_logistic_lasso"),
    args = list(
      iobject = expr(object$fit),
      new_x = expr(as.matrix(new_data[, names(object$fit$beta)])))
  )
)
