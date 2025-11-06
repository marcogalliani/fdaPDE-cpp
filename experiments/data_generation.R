source("data/data_generator.R")
source("data/mesh_utils.R")

data_path <- "data/fpca-na/"
mesh_path <- "mesh/unit_square/"

## Mesh ----
n_nodes <- 400
mesh <- unit_square.mesh(n_nodes, mesh_path)

## Sampling locations ----
n_locs <- 900
locs <- unit_square.locs(n_locs, data_path)

# computing the Psi matrix
Vh <- FunctionSpace(mesh, fe_order=1)
u <- Function(Vh)
Lu <- -laplace(u) ## poisson problem
f <- function(points) { return(0*points[,1])}
pde <- Pde(Lu, f)
Psi <- Vh$basis()$eval(as.matrix(locs))

## Mean ----
log_mean_generator <- function(locs) {
  ## I want a function between -1 and 1
  return((2 * log((locs[, 1] + locs[, 2]) + 1) / log(3) - 1))
}
mean_locs <- log_mean_generator(locs)
plot_eigenfunction(mean_locs,locs)

## Eigenfunctions ----
n_pcs <- 3
laplacian_square_eigenfunction <- function(a, b, locs, x_t=0.2,y_t=0) {
  return(cos(a * pi * (locs[,1]-x_t)) * cos(b * pi * (locs[, 2]-y_t)))
}

ab_pairs <- list()
a <- 1
b <- 1
for(i in 1:n_pcs){
  ab_pairs[[i]] <- c(a,b)
  #if(i%%2==0){
    a <- a + 1
  #}else{
    b <- b + 1
  #}
}

## Compute and bind results
test_functions <- do.call(cbind, lapply(ab_pairs, function(ab) {
  laplacian_square_eigenfunction(a = ab[1], b = ab[2], locs = mesh$nodes())
}))
test_functions <- apply(test_functions, MARGIN=2, function(f,R0){ return(f/ sqrt(as.numeric(t(f) %*% R0 %*% f)))}, pde$mass())

test_functions_locs <- Psi %*% test_functions

pc_idx <- 1
while(pc_idx < ncol(test_functions)){
  ab_pairs[[pc_idx]]
  print(plot_eigenfunction(test_functions_locs[,pc_idx],locs))
  pc_idx <- pc_idx + 1
}


## Data ----
n_units <- 50
scores_sd <- seq(from=0.4,to=0.2, length.out=n_pcs)
NSR <- 1
error_sd  <- sqrt(NSR * sum(scores_sd^2))

data <- data_generator(
  n_units,
  true_PCs = as.matrix(test_functions),
  Psi = Psi,
  scores_sd = scores_sd,
  error_sd = error_sd,
  obs_path = data_path,
  mean_locs = mean_locs,
  seed = 1412,
  filename = "y.csv")


## Add missingness
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
                              p = 0.5, 
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


## 
partial_data <- t(apply(data$noisy_data, 1, 
                        dep_space.NA_fast,
                        p=0.5,
                        schema="c",
                        mesh_ref=mesh,
                        locations=locs,
                        RDD_groups=12,
                        w=8))

## Write data
write.csv(format(partial_data, digits = 16),
          paste0(data_path,"y.csv"))

write.csv(format(locs, digits = 16),
          paste0(data_path,"locs.csv"))

write.csv(format(as.matrix(data$noisy_data), digits = 16),
          paste0(data_path,"y_complete.csv"))



