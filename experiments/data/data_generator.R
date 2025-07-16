data_generator <- function(n_units, true_PCs, Psi, scores_sd, error_sd, mean_locs = NULL, seed=42, obs_path=NULL, filename = "y.csv"){
  set.seed(seed)
  data <- list()
  data_range <- max(true_PCs) - min(true_PCs)
  
  data$rank <- ncol(true_PCs)
  data$pcs <- true_PCs
  data$Psi <- Psi

  # scores & exact data matrix
  data$scores <- matrix(nrow=n_units, ncol=ncol(true_PCs))
  for(i in 1:ncol(true_PCs)){
    data$scores[,i] <- rnorm(n = n_units, sd = scores_sd[i] * data_range)
    data$scores[,i] <- data$scores[,i] - mean(data$scores[,i])
  }
  data$X_true <- data$scores %*% t(Psi %*% data$pcs)
  
  if(!is.null(mean_locs)){
    data$center_locs <- mean_locs
    data$X_true <- data$X_true + rep(1, n_units) %*% t(mean_locs)
  }

  
  #compute the noisy data matrix
  error <- rnorm(n = n_units*nrow(Psi), sd = error_sd*data_range)
  error <- matrix(error,nrow=n_units)
  error <- error - matrix(rep(colMeans(error),n_units), byrow=T, nrow=n_units)
  
  data$noisy_data <- as.matrix(data$X_true + error)
  data$error <- error
  
  
  # add a heteroschedastic error
  
  
  #(4) save the noisy matrix
  if(!is.null(obs_path)){
    write.csv(format(as.matrix(data$noisy_data), digits = 16),
              paste(obs_path,filename,sep="/"))
  }
  return(data)
}