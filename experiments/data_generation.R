source("data/data_generator.R")
source("/Users/marcogalliani/Projects/fdaPDE-cpp/experiments/data/mesh_utils.R")

data_path <- "data/fpca-fully_obs"
mesh_path <- "data/mesh/unit_square"

## Mesh ----
n_nodes <- 400
mesh <- unit_square.mesh(n_nodes, mesh_path)

## Sampling locations ----
n_locs <- 600
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
laplacian_square_eigenfunction <- function(a, b, locs, x_t=0.2,y_t=0.2) {
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
  filename = "y.csv"
)

NSR <- error_sd^2/sum(scores_sd^2)
NSR

## Variance ----
# X = SF^T + E
data_range <- max(test_functions) - min(test_functions)
true_total_var <- sum(c(scores_sd,error_sd)^2)*data_range^2
true_total_var

true_pcs_var <- sum(scores_sd^2)*data_range^2
true_pcs_var

scores_sd^2/sum(c(scores_sd,error_sd)^2)
true_total_var

centred_data <- sweep(data$noisy_data, 2, colMeans(data$noisy_data), FUN = "-")
estimated_total_var <- 1/(n_units-1)*sum(centred_data^2)
estimated_total_var

apply(data$scores,2,var) * n_locs/ (estimated_total_var)

estimated_pcs_var <- 1/(n_units-1)*sum((data$scores%*%t(Psi%*%data$pcs))^2)
estimated_pcs_var

plot(as.factor(1:n_pcs), scores_sd^2, type="l")
plot(1:n_pcs, scores_sd^2/sum(c(scores_sd,error_sd)^2))


