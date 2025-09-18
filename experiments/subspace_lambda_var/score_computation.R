source("data_generation.R")

script_path <- "subspace_lambda_var/"
## Fit ----
# specify the params within the json file
library(jsonlite)
params_json <- fromJSON(paste(script_path,"params.json",sep=""))

lambda_grid <- 10^seq(2,4,by=0.5)

params_json$RunParams$fpca_solver <- "subspace_experimental"
params_json$RunParams$n_pc <- n_pcs
params_json$RunParams$lambda_grid <- lambda_grid
write_json(path = paste(script_path,"params.json",sep=""), 
           params_json, auto_unbox=T, pretty=T, digits=10
)

# fit the model by running the program
system(paste("cd ", script_path,"; ","./subspace_it",sep=""),
       ignore.stdout = T)


## Results ----
loadings_locs <- as.matrix(read.csv(paste(script_path,"test-results/loadings_locs.csv",sep="")))
loadings <- as.matrix(read.csv(paste(script_path,"test-results/loadings.csv",sep="")))
scores <- as.matrix(read.csv(paste(script_path,"test-results/scores.csv",sep="")))
gcv_scores <- read.csv(paste(script_path,"test-results/gcv_scores.csv",sep=""))

center_locs <- read.csv(paste(script_path,"test-results/center_locs.csv",sep=""))

## compute the scores in an alternative way:
scores_proj <- t(solve(a = t(loadings_locs)%*%loadings_locs, b = t((data$noisy_data-rep(1,n_units)%*%t(center_locs)) %*% loadings_locs)))

mean((data$X_true - scores_proj %*% t(loadings_locs))^2)
mean((data$X_true - scores %*% t(loadings_locs))^2)


align_matrices <- function(m_ref, m_var){
  col_prods <- apply(m_ref*m_var,2,sum)
  change_sign <- sign(col_prods)
  change_sign <- matrix(rep(change_sign,nrow(m_ref)),
                     ncol = ncol(m_ref),
                     byrow = T)
  m_var * change_sign
}

aligned_scores <- align_matrices(data$scores, scores)
aligned_scores_proj <- align_matrices(data$scores, scores_proj)

print("not aligned")
apply(data$scores - scores_proj, 2, function(v){mean(v^2)})
apply(data$scores - scores, 2, function(v){mean(v^2)})

print("aligned")
apply(data$scores - aligned_scores_proj, 2, function(v){mean(v^2)})
apply(data$scores - aligned_scores, 2, function(v){mean(v^2)})




