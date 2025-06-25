# Generating partially observed fPCA data on the unit square ----

setwd("/Users/marcogalliani/Projects/fdaPDE-cpp/experiments/partially-observed")
library(fdaPDE)
library(femR)

source("/Users/marcogalliani/Projects/fdaPDE-cpp/experiments/data/mesh_utils.R")

datadir = "../data/fpca/" # even "002/" and "003/"

## Unit square mesh ----
meshdir = "../data/mesh/unit_square/"


mesh <- unit_square.mesh(400,meshdir)

## Eigenfunctions ----

#' The PC orthogonal functions used to generate the data are the eigenfunctions
#' of the laplacian operator with Neumann bdd conditions
laplacian_square_eigenfunction <- function(a, b, locs, x_t=0,y_t=0) {
  return(cos(a * pi * (locs[,1]-x_t)) * cos(b * pi * (locs[, 2]-y_t)))
}
f1_fem <- laplacian_square_eigenfunction(a = 1, b = 1, mesh$nodes())
f2_fem <- laplacian_square_eigenfunction(a = 2, b = 3, mesh$nodes())
f3_fem <- laplacian_square_eigenfunction(a = 4, b = 4, mesh$nodes())
test_functions <- cbind(f1_fem,f2_fem,f3_fem)

#' The functions are normalized to have unit norm (w.r.t. R0)
Vh <- FunctionSpace(mesh, fe_order=1)
u <- Function(Vh)
Lu <- -laplace(u) ## poisson problem
f <- function(points) { return(0*points[,1])}
pde <- Pde(Lu, f)
test_functions <- apply(test_functions, MARGIN=2, function(f,R0){ return(f/ sqrt(as.numeric(t(f) %*% R0 %*% f)))}, pde$mass())

#' Then the functions are evaluated at locations
locations <- unit_square.locs(400,datadir)
test_functions.at_locs <- Vh$basis()$eval(as.matrix(locations)) %*% test_functions

## Generate the data----
data_range <- max(test_functions.at_locs) - min(test_functions.at_locs)
n_units <- 50
scores_sd <- c(0.4,0.3,0.2)
# (1) scores & exact data matrix
scores <- matrix(nrow=n_units, ncol=ncol(test_functions.at_locs))
for(i in 1:ncol(test_functions.at_locs)){
  scores[,i] <- rnorm(n = n_units, sd = scores_sd[i] * data_range)
  scores[,i] <- scores[,i] - mean(scores[,i]) # centering
}
#(2) compute the noisy data matrix
error_sd <- 0.1
error <- rnorm(n = n_units*nrow(test_functions.at_locs), sd = error_sd*data_range)
error <- matrix(error, nrow=n_units)
error <- error - matrix(rep(colMeans(error), n_units), byrow=T, nrow=n_units) # centering

noisy_data <- scores %*% t(test_functions.at_locs) + error


ind.space.NA <- function(data_vector,p=0.75){
  size <- length(data_vector)
  data_vector[sample(1:size,size-round(size*p))] <- NA
  return(data_vector)
}

## missingness
library(RANN)

# Optimized distance function
de <- function(x, y, w = 1) {
  res <- sum((x[1:2] - y[1:2])^2)
  if (length(x) == 3) {
    res <- res + w^2 * (x[3] - y[3])^2
  }
  sqrt(res)
}

# Optimized nearest neighbor function
nearest <- function(p, points, w = 1) {
  distances <- apply(points, 1, de, y = p, w = w)
  which.min(distances)
}

dep_space.NA_fast <- function(Data, 
                              p = 0.75, 
                              schema = "a", 
                              mesh_ref = NULL, 
                              locations = NULL, 
                              RDD_groups = NULL, 
                              w = 8) {
  if (schema == "c") {
    data_vector <- as.vector(Data)
    size <- length(data_vector)
    
    nodes <- mesh_ref$nodes()
    nobs <- ceiling(RDD_groups * p)
    pts <- locations
    
    # Use precomputed seeds if possible
    seeds <- nodes[sample(nrow(nodes), RDD_groups), , drop = FALSE]
    obs_marker <- rep(0, RDD_groups)
    obs_marker[sample(RDD_groups, nobs)] <- 1
    
    # Fast nearest neighbor search using RANN
    nn_results <- nn2(seeds, pts, k = 1)  # Finds nearest seed for each location
    nearest_idxs <- nn_results$nn.idx  # Index of the nearest seed
    
    # Vectorized NA assignment
    data_vector[obs_marker[nearest_idxs] == 0] <- NA
    
    Data <- matrix(data = data_vector, nrow = 1, ncol = size)
  }
  
  Data
}


#partial_data <- t(apply(noisy_data, 1, ind.space.NA,0.5))
partial_data <- t(apply(noisy_data, 1, 
        dep_space.NA_fast,
        p=0.5,
        schema="c",
        mesh_ref=mesh,
        locations=locations,
        RDD_groups=12,
        w=8))

write.csv(format(partial_data, digits = 16),
          paste(datadir,"y.csv",sep=""))

write.csv(format(locations, digits = 16),
          paste(datadir,"locs.csv",sep=""))

write.csv(format(as.matrix(noisy_data), digits = 16),
          paste(datadir,"y_complete.csv",sep=""))


