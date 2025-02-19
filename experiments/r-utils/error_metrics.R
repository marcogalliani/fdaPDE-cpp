library(pracma)
library(Matrix)

RMSE.vector <- function(v1,v2){
  return(sqrt(mean((v1-v2)^2)))
}

RMSE.matrix <- function(m1,m2){
  return(norm(m1-m2,type="F")/sqrt(ncol(m1)*nrow(m1)))
}

compute.errors.FPCA <- function(
    data, est.PCs, est.Scores){
  
  #compute errors & times
  errors <- list()
  errors$RMSE.PCs <- vector()
  errors$Scores <- vector()
  
  for(ind_pc in 1:min(ncol(est.PCs),ncol(data$pcs))){
    errors$RMSE.PCs[ind_pc] <- 
      min(RMSE.vector(data$pcs[,ind_pc],est.PCs[,ind_pc]),
          RMSE.vector(data$pcs[,ind_pc],-est.PCs[,ind_pc]))
    errors$Scores[ind_pc] <- 
      min(RMSE.vector(data$scores[,ind_pc],est.Scores[,ind_pc]),
          RMSE.vector(data$scores[,ind_pc],-est.Scores[,ind_pc]))
  }
  
  errors$signal_reconstruction <- RMSE.matrix(data$scores%*%t(data$pcs),
                                              est.Scores%*%t(est.PCs))
  errors$space_reconstruction <- subspace(data$pcs,est.PCs)
  
  return(errors)
}
