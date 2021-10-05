
coord_descent = function(x, y, beta, lambda, weights, working_responses){
  
  ## coord_descent completes one iteration of the 
  ## coordinate descent algorithm on all parameters 
  ## to minimize the penalized weighted least squares of the
  ## logistic model. Serves as a helper function for fitting a
  ## logistic lasso.
  ## 
  ## Input:
  ## - x: a numeric matrix where the first column contain 1's
  ##      and the subsequent columns are the regressor values
  ##      (the design matrix).
  ## - y: a numeric vector of length = nrow(x) containing values
  ##      of the logit binary response variable.
  ## - beta: a numeric vector of length = ncol(x) which specifies
  ##         the initial guess of the intercept and coefficients.
  ## - lambda: numeric value for the penalty
  ## - weights: a numeric vector of the weights for the WLS problem
  ## - working_responses: a numeric vector of the working responses
  ##
  ## Output:
  ## - Returns a numeric vector of updated parameters from a single
  ##   iteration of coordinate descent.
  ##
  ## Example:
  ##   library(tidyverse)
  ##   ncol(x) == 4 #TRUE
  ##   CD_output = coord_descent(x, y, beta = c(0,0,0,0), lambda = 0.1,
  ##                             weights = weights, working_response = WR)
  
  nrow = dim(x)[1]; ncol = dim(x)[2]
  wmat = diag(as.vector(weights))
  
  for (j in 1:ncol){
    
    r = working_responses - x[,-j] %*% beta[-j] # compute residuals
    
    if (j == 1){
      # unconstrained optimization for intercept
      beta[j] = lm(r ~ 1, weights = weights) %>% coef()
    }
    else{
      # update beta using formula derived for
      # univariate logistic LASSO (with weights)
      xw = t(x[,j]) %*% wmat
      xwr = xw %*% r
      xwx = xw %*% x[,j]
      beta[j] = sign(xwr) * max(abs(xwr) - lambda*nrow, 0) / (xwx)
    }
  } # end iteration through 1:ncol
  
  return(beta)
}# end function




fit_logistic_lasso = function(x, y, lambda, beta0 = NULL, 
                              eps = 0.001, iter_max = 100){
  
  ## fit_logistic_lasso uses the coordinate descent algorithm
  ## to compute the Iteratively Reweighted Least Squares problem
  ## of the logistic lasso regression.
  ## 
  ## Input:
  ## - x: a dataframe or matrix where the columns are the regressor values.
  ## - y: a factor vector of length = nrow(x) containing values
  ##      of the logit binary response variable.
  ## - lambda: numeric value for the penalty
  ## - beta0: a numeric vector of length = ncol(x) + 1 which specifies
  ##          the initial guess of the intercept and coefficients.
  ## - eps: a numeric value that determines the convergence threshold.
  ## - iter_max: the maximum number of iterations of reweighted least squares
  ##             before halting the algorithm.
  ##
  ## Output:
  ## - Returns a list of three components: (1) numeric value of the estimated 
  ##   intercept, (2) a numeric vector of the estimated coefficients, and
  ##   (3) the numeric value of the penalty.
  ##
  ## Example:
  ##   library(tidyverse); library(tidymodels)
  ##   ncol(x) == 3 #TRUE
  ##   fit = fit_logistic_lasso(x, y, lambda = 0.1, beta = c(0,0,0,0), 
  ##                            eps = 0.001, iter_max = 200) 
  
  x = data.matrix(x) # format input as matrix
  x = cbind(1, x) # format design matrix
  ncol = dim(x)[2]
  
  # store levels of y and convert to numeric 0/1
  fct_levels = levels(y)
  y = as.numeric(y) - 1
  
  if(!is.null(beta0)){
    beta = beta0 # duplicate beta0 for updating
  } 
  else {
    beta0 = rep(0, ncol)
    beta = beta0 # duplicate beta0 for updating
  }
  
  names(beta) = colnames(x); names(beta)[1] = "Intercept"
  
  
  for (iter in 1:iter_max){ # BEGIN OUTER ITERATION
    
    beta_last_l1 = beta # store copy of beta
    x_beta = as.numeric(x %*% beta)
    p = 1/(1 + exp(-x_beta)) # compute probabilities with logistic function
    weights = p*(1-p)
    working_responses = (y - p)/weights + x_beta
    
    
    for (subiter in 1:iter_max){ # BEGIN SUB-ITERATION
      
      beta_last_l2 = beta
      beta = coord_descent(x, y, beta, lambda, weights, working_responses)
      inner_conv = subiter
      
      if ( max(abs(beta - beta_last_l2)) < eps ){ # convergence of sub-iteration
        #cat(paste("Iteration", iter, "subiteration", subiter), "\r")
        break}
      if (subiter == iter_max){
        return(list(success = FALSE, iter = iter, beta_fail = beta))}
    } # END SUB-ITERATION
    
    
    if ( max(abs(beta - beta_last_l1)) < eps ){ # convergence of iteration
      #print(paste("Converged on iteration", iter))
      return(list(success = TRUE, intercept = beta[1], beta = beta[-1], 
                  lambda = lambda, factor_lvls = fct_levels))}
  } # END OUTER ITERATION
  
  warning("Convergence took more than iter_max iterations")
  return(list(success = FALSE, iter = iter_max, beta_fail = beta))
} # close function




predict_logistic_lasso = function(iobject, new_x){
  
  ## predict_logistic_lasso classifies the input datapoints
  ## into 0 or 1 (which corresponds to the two levels of the outcome 
  ## variable). Classification is based on the fitted model using
  ## fit_logistic_lasso.
  ##
  ## Input:
  ## - iobject: the output of fit_logistic_lasso.
  ## - new_x: a numeric matrix where each column is
  ##          a regressor and each row is a data point.
  ## Output:
  ## - Returns a binary factor vector of the classified points.
  ##
  ## Example:
  ##   library(tidyverse); library(tidymodels)
  ##   ncol(x) == 3 #TRUE
  ##   fit = fit_logistic_lasso(x, y, lambda = 0.1, beta = c(0,0,0,0), 
  ##                            eps = 0.001, iter_max = 200)
  ##   ncol(new_x) == 3 #TRUE
  ##   predict_logistic_lasso(fit, new_x)
  
  intercept = iobject$intercept
  beta = iobject$beta
  x_beta = intercept + new_x %*% beta
  
  # Bayesian classifier decision boundary
  predict = ifelse(x_beta > 0, 1, 0) %>% as.numeric()
  
  return(iobject$factor_lvls[predict+1] %>% factor())
}

