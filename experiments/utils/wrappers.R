#' Wrapper for C++-based fPCA models
#' This function runs the external C++ executable and reads/formats the results
#' into a standardized list.
run_fpca_na <- function(partial_data, model_name, data_path) {
  ## write data to cpp solver
  write.csv(format(as.matrix(partial_data), digits = 16),
            paste0(data_path, "y.csv"))
  
  ## Initialize empty model
  model <- list()
  cpp_path <- paste0(model_name, "/")
  results_path <- paste0(cpp_path, "test-results/")
  
  # Run C++ executable ----
  start.time <- Sys.time()
  system(paste0("cd ", model_name, " && ", "./fit_model"),
         ignore.stdout = F)
  end.time <- Sys.time()
  
  # Load results ----
  results <- list()
  results$center_locs <- as.matrix(read.csv(paste0(results_path, "center_locs.csv")))
  results$loadings_locs <- as.matrix(read.csv(paste0(results_path, "loadings_locs.csv")))
  results$scores <- as.matrix(read.csv(paste0(results_path, "scores.csv")))
  results$X_hat_locs <- as.matrix(read.csv(paste0(results_path, "reconstruction_at_locs.csv")))
  results$cv_scores <- as.matrix(read.csv(paste0(results_path, "gcv_scores.csv")))
  results$execution_time <- end.time - start.time
  
  # Standardize output
  return(list(
    method = model_name,
    rank = ncol(results$loadings_locs),
    center_locs = as.vector(results$center_locs),
    loadings_locs = results$loadings_locs,
    scores = results$scores,
    X_hat_locs = results$X_hat_locs, # Smoothed reconstruction
    X_imputed = results$X_hat_locs,  # For fpca, imputation = reconstruction
    execution_time = results$execution_time
  ))
}

#' Wrapper for DINEOF
#' This function runs sinkr::dineof, performs PCA on the result, and
#' formats the output into a standardized list.
run_dineof <- function(partial_data, n_comp) {
  start.time <- Sys.time()
  dineof_fit <- sinkr::dineof(partial_data)
  pca_fit <- wrapped_pca(dineof_fit$Xa, center = TRUE, n_comp = n_comp)
  end.time <- Sys.time()
  
  # Standardize output
  return(list(
    method = "dineof",
    rank = dineof_fit$n.eof,
    center_locs = as.vector(pca_fit$results$X_mean_locs),
    loadings_locs = pca_fit$results$loadings_locs,
    scores = pca_fit$results$scores,
    X_hat_locs = pca_fit$results$X_hat_locs, # Smoothed reconstruction
    X_imputed = dineof_fit$Xa,              # DINEOF's imputation
    execution_time = end.time - start.time
  ))
}

#' Wrapper for SoftImpute
#' This function runs softImpute::softImpute and formats the output
#' into a standardized list.
#' It uses cv.softImpute to automatically select the best lambda.

library(RMCLab)
run_soft_impute <- function(partial_data, n_pc, rank = 50, lambda = 30) {
  start.time <- Sys.time()
  
  calibrate_soft_impute <- soft_impute_tune(partial_data, 
                                            lambda = fraction_grid(nb_lambda = 6, 
                                                                   reverse = TRUE),
                                            splits = cv_folds_control(K = 5L))
  
  
  end.time <- Sys.time()
  
  # Reconstruct results from the fit at lambda.min
  # cv_fit itself is a softImpute object fit at lambda.min
  loadings_at_locs <- calibrate_soft_impute$fit$svd$v
  scores <- calibrate_soft_impute$fit$svd$u %*% diag(calibrate_soft_impute$fit$svd$d)
  X_reconstructed <- calibrate_soft_impute$fit$X
  
  # Standardize output
  return(list(
    method = "soft_impute",
    rank = length(calibrate_soft_impute$fit$sv$d), # The rank at the optimal lambda
    # softImpute centers, but doesn't return the mean.
    # We follow the original script's convention of using 0 for the metric.
    center_locs = rep(0, ncol(partial_data)), 
    loadings_locs = loadings_at_locs,
    scores = scores,
    X_hat_locs = X_reconstructed, # Smoothed reconstruction
    X_imputed = X_reconstructed,  # Imputation
    execution_time = end.time - start.time
  ))
}


#' General model-fitting dispatcher
#' This function calls the appropriate wrapper based on the model name.
fit_model <- function(model_name, partial_data, data_path, n_pc) {
  switch(model_name,
         fpca_na_gcv = {
           return(run_fpca_na(partial_data, "fpca-na-gcv", data_path))
         },
         dineof = {
           return(run_dineof(partial_data, n_pc))
         },
         soft_impute = {
           # Using default params from original loop
           return(run_soft_impute(partial_data, n_pc, rank = 50, lambda = 30))
         },
         {
           stop(paste("The model", model_name, "does not exist"))
         }
  )
}