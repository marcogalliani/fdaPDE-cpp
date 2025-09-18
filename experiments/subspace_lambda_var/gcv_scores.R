source("data_generation.R")

script_path <- "subspace_lambda_var/"
## Fit ----
# specify the params within the json file
library(jsonlite)
params_json <- fromJSON(paste(script_path,"params.json",sep=""))

lambda_grid <- 10^seq(-9,4,by=0.5)

params_json$RunParams$fpca_solver <- "subspace_experimental"
params_json$RunParams$n_pc <- n_pcs
params_json$RunParams$lambda_grid <- lambda_grid
write_json(path = paste(script_path,"params.json",sep=""), 
           params_json, auto_unbox=T, pretty=T, digits=10
)

# fit the model by running the program
system(paste("cd ", script_path,"; ","./subspace_it",sep=""),
       ignore.stdout = T)


plot_stat_units(data$noisy_data[1,],locs)

## Results ----
gcv_scores <- read.csv(paste(script_path,"test-results/gcv_scores.csv",sep=""))
edf_scores <- read.csv(paste(script_path,"test-results/edf_scores.csv",sep=""))
loadings_locs <- as.matrix(read.csv(paste(script_path,"test-results/loadings_locs.csv",sep="")))
scores <- as.matrix(read.csv(paste(script_path,"test-results/scores.csv",sep="")))

### GCV curves ----
data_fidelity <- sum((t(data$noisy_data)%*%scores-loadings_locs)^2)

# Sort lambda_grid (if not already sorted)
ord <- order(lambda_grid)
lambda_grid_sorted <- lambda_grid[ord]
gcv_scores_sorted <- gcv_scores[ord, ]

# Set up colors (one per curve)
num_curves <- ncol(gcv_scores_sorted)
curve_colors <- rainbow(num_curves)

# Plot GCV curves on log-log scale
matplot(lambda_grid_sorted, gcv_scores_sorted, type = "l", log = "xy",
        xlab = expression(lambda), ylab = "GCV Score",
        lty = 1, col = curve_colors, lwd = 2)

# Find minimum GCV scores and corresponding lambdas
min_indices <- apply(gcv_scores_sorted, 2, which.min)
min_lambdas <- lambda_grid_sorted[min_indices]
min_scores <- apply(gcv_scores_sorted, 2, min)

# Add points at minima
matpoints(min_lambdas, min_scores, pch = 21, bg = "white", col = curve_colors, cex = 1.5)
matlines(lambda_grid, apply(gcv_scores,1,sum)/n_pcs, col = "orange")


## Add gcv scores for unique lambda, for reference
params_json$RunParams$fpca_solver <- "subspace"
write_json(path = paste(script_path,"params.json",sep=""), 
           params_json, auto_unbox=T, pretty=T, digits=10
)
system(paste("cd ", script_path,"; ","./subspace_it",sep=""),
       ignore.stdout = T)
gcv_scores_reference <- read.csv(paste(script_path,"test-results/gcv_scores.csv",sep=""))

matlines(lambda_grid, gcv_scores_reference/n_pcs, col = "black", cex = 1.5)

# Add the direct method for comparison
## Add gcv scores for unique lambda, for reference
params_json$RunParams$fpca_solver <- "direct"
write_json(path = paste(script_path,"params.json",sep=""), 
           params_json, auto_unbox=T, pretty=T, digits=10
)
system(paste("cd ", script_path,"; ","./subspace_it",sep=""),
       ignore.stdout = T)
gcv_scores_direct <- read.csv(paste(script_path,"test-results/gcv_scores.csv",sep=""))

# matplot(lambda_grid_sorted, gcv_scores_direct, type = "l", log = "xy",
#         xlab = expression(lambda), ylab = "GCV Score",
#         lty = 1, col = curve_colors, lwd = 2)

matlines(lambda_grid, gcv_scores_direct/n_pcs, col = "purple", cex = 1.5)

## Add a reconstruction error curve
uncal_script_path <- "uncalibrated-fpca/"
uncal_json <- fromJSON(paste(uncal_script_path,"params.json",sep=""))

uncal_json$RunParams$fpca_solver <- "direct"
uncal_json$RunParams$n_pc <- n_pcs
  
rec_error <- numeric(length(lambda_grid_sorted))
for(i in 1:length(lambda_grid_sorted)){
  uncal_json$RunParams$lambda_grid <- I(lambda_grid_sorted[i])
  
  write_json(path = paste(uncal_script_path,"params.json",sep=""), 
             uncal_json, auto_unbox=T, pretty=T, digits=10
  )
  
  system(paste("cd ", uncal_script_path,"; ","./uncalibrated_fpca",sep=""),
         ignore.stdout = T)
  
  X_hat <- as.matrix(read.csv(paste(uncal_script_path,"test-results/reconstruction_at_locs.csv",sep="")))
  mean_locs <- as.matrix(read.csv(paste(uncal_script_path,"test-results/center_locs.csv",sep="")))
  
  rec_error[i] <- mean((data$X_true-X_hat)^2)
}
  
matlines(lambda_grid, rec_error/n_pcs, col = "pink", cex = 1.5)

# Add a legend
legend("topleft", legend = c(paste("fPC", 1:num_curves, sep=""),"Sum","Unique param","direct","true"),
       col = c(curve_colors,"orange","black","purple","pink"), lty = 1, lwd = 2, bty = "n", title = "GCV Curves")

### true error ----
matplot(lambda_grid_sorted, cbind(rec_error,gcv_scores_reference/n_pcs), type = "l", log = "xy",
        xlab = expression(lambda), ylab = "reconstruction",
        lty = 1, col = c("pink","black"), lwd = 2)

# Find minimum GCV scores and corresponding lambdas
min_indices <- apply(gcv_scores_reference, 2, which.min)
min_lambdas <- lambda_grid_sorted[min_indices]
min_scores <- apply(gcv_scores_reference, 2, min)/3

# Add points at minima
abline(v = min_lambdas)


### edf curves ----
matplot(lambda_grid_sorted, edf_scores, type = "l", log = "xy",
        xlab = expression(lambda), ylab = "edf Score",
        lty = 1, col = "red", lwd = 2)



### data fidelity curves ----
residual_dof <- (n_locs-edf_scores$V0)^2
data_fidelity <- 
  sweep(gcv_scores/n_locs, 1,
        residual_dof,
        FUN="*")

matplot(lambda_grid_sorted, data_fidelity, type = "l", log = "xy",
        xlab = expression(lambda), ylab = "Data fidelity",
        lty = 1, col = curve_colors, lwd = 2)
NSR
# Add a legend
legend("topleft", legend = paste("fPC", 1:num_curves, sep=""),
       col = curve_colors, lty = 1, lwd = 2, bty = "n", title = "GCV Curves")

matplot(lambda_grid_sorted, subspace_data_fidelity, type = "l", log = "xy",
        xlab = expression(lambda), ylab = "GCV Score",
        lty = 1, col = curve_colors, lwd = 2)


## Check with the old library ----
# library(fdaPDE)
# mesh_fdapde <- create.mesh.2D(nodes = mesh$nodes())
# FEMbasis <- create.FEM.basis(mesh = mesh_fdapde)
# 
# solution <- FPCA.FEM(locations = locs,
#                      datamatrix = data$noisy_data,
#                      FEMbasis = FEMbasis,
#                      validation = "GCV",
#                      lambda = lambda_grid, #grid of possible lambdas
#                      nPC = 3)
# 
# 
# loadings <- read.csv(paste(script_path,"test-results/loadings.csv",sep=""))
# 
# plot_stat_units(data$pcs[,3],mesh$nodes())
# plot_stat_units(loadings[,3],mesh$nodes())
# plot_stat_units(solution$loadings.FEM$coeff[,3],mesh$nodes())



