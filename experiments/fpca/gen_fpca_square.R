library(fdaPDE)
library(femR)

mesh_data_path <- "data/mesh/unit_square_rsvd_test"

## define eigenfunctions of laplace operator over square with neumann boundary conditions
square_eigenfunction <- function(a, b, locs) {
    return(cos(a * pi * locs[,1]) * cos(b * pi * locs[, 2]))
}

## define spatial domain
N <- 50 ## number of nodes in spatial domain
x.2D <- seq(0, 1, length.out = N)
y.2D <- x.2D
locations.2D <- expand.grid(x.2D, y.2D)
mesh.2D <- fdaPDE::create.mesh.2D(locations.2D)

## export mesh
write.csv(format(mesh.2D$nodes, digits = 16), paste(mesh_data_path,"points.csv", sep = "/"))
write.csv(format(mesh.2D$triangles, digits = 16), paste(mesh_data_path,"elements.csv", sep = "/"))
write.csv(format(1 * mesh.2D$nodesmarkers, digits = 16), paste(mesh_data_path,"boundary.csv", sep = "/"))
write.csv(format(mesh.2D$neighbors, digits = 16), paste(mesh_data_path,"neigh.csv", sep = "/"))
write.csv(format(mesh.2D$edges, digits = 16), paste(mesh_data_path,"edges.csv", sep = "/"))

## Observation data
obs_data_path <- "data/models/fpca/2D_test_rsvd"

## define observations' locations
n <- 50 ## number of observation locations
x.obs.2D <- seq(0, 1, length.out = n)
y.obs.2D <- x.obs.2D
locations.obs.2D <- expand.grid(x.obs.2D, y.obs.2D)
n_locations <- nrow(locations.obs.2D)

## export locations
write.csv(format(locations.obs.2D, digits = 16), paste(obs_data_path,"locs.csv", sep = "/"))

plot(mesh.2D)
points(locations.obs.2D, col = "red")

## use n = N for the setting locations = mesh nodes
## (in this case, \Psi is the identity matrix)
## use n != N for the general case. Set n > N.

## L2 norm

## use femR to extract mass matrix
mesh <- Mesh(
  list(
    nodes = mesh.2D$nodes,
    elements = mesh.2D$triangles,
    boundary = mesh.2D$nodesmarkers
  )
)
## define a dummy PDE, just to get the mass matrix
Vh <- FunctionSpace(mesh, fe_order=1)
u <- Function(Vh)
Lu <- -laplace(u) ## poisson problem
f <- function(points) { return(0*points[,1])}
pde <- Pde(Lu, f)

R0 <- pde$mass() ## mass matrix

L2norm <- function(f) { return(sqrt(as.numeric(t(f) %*% R0 %*% f))) }

## evaluate eigenfunctions on mesh nodes (this is equivalent to get the FEM basis
## expansion of the eigenfunctions ). change values of a and b to select eigenfunction
f1_fem <- square_eigenfunction(a = 1, b = 1, mesh.2D$nodes)
f2_fem <- square_eigenfunction(a = 2, b = 3, mesh.2D$nodes)
f3_fem <- square_eigenfunction(a = 4, b = 4, mesh.2D$nodes)

t(f1_fem) %*% R0 %*% f1_fem

## normalize
f1_norm <- L2norm(f1_fem)
f2_norm <- L2norm(f2_fem)
f3_norm <- L2norm(f3_fem)

f1_normalized <- f1_fem / f1_norm
f2_normalized <- f2_fem / f2_norm
f3_normalized <- f3_fem / f3_norm

## obervations are evaluation of L^2 normalized eigenfunctions at data locations
Psi <- Vh$basis()$eval(as.matrix(locations.obs.2D))

f1 <- as.matrix(Psi %*% f1_normalized)
f2 <- as.matrix(Psi %*% f2_normalized)
f3 <- as.matrix(Psi %*% f3_normalized)


## plot
library(ggplot2)
library(viridis)

plot_eigenfunction <- function(data, locs) {
    ## define data.frame for plotting
    df <- data.frame(
        x = locs[,1],
        y = locs[,2],
        f = data
    )
    ## colors
    n_breaks <- 50
    mybreaks <- c(-Inf, seq(min(data), max(data), length.out = n_breaks), Inf)
    mycolors<- function(x) {
        colors<-colorRampPalette(viridis(11))( x + 1 )
        colors[1:x]
    }
    ## plot
    p <- ggplot() +
        geom_contour_filled(data = df, aes(x, y, z = f),
                            breaks = mybreaks) +
        scale_fill_manual(
            aesthetics = "fill",
            values = mycolors(n_breaks + 2), name = "Value", drop = FALSE
        ) +
        coord_equal() + 
        theme_void() +
        theme(legend.position = "none")
    p
}
plot_eigenfunction(f1, locations.obs.2D)
plot_eigenfunction(f2, locations.obs.2D)
plot_eigenfunction(f3, locations.obs.2D)

## define eigenfunctions of laplace operator over square with neumann boundary conditions
translated_eigenfunction <- function(a, b, locs,x_translation, y_translation) {
  return(cos(a * pi * (locs[,1]-x_translation)) * cos(b * pi * (locs[, 2]-y_translation)))
}

## evaluate eigenfunctions on mesh nodes (this is equivalent to get the FEM basis
## expansion of the eigenfunctions ). change values of a and b to select eigenfunction
f1_fem <- translated_eigenfunction(a = 1, b = 1, mesh.2D$nodes,0.1,0.1)
f2_fem <- translated_eigenfunction(a = 2, b = 3, mesh.2D$nodes,0.1,0.1)
f3_fem <- translated_eigenfunction(a = 4, b = 4, mesh.2D$nodes,0.1,0.1)

## normalize
f1_norm <- L2norm(f1_fem)
f2_norm <- L2norm(f2_fem)
f3_norm <- L2norm(f3_fem)

f1_normalized <- f1_fem / f1_norm
f2_normalized <- f2_fem / f2_norm
f3_normalized <- f3_fem / f3_norm

## obervations are evaluation of L^2 normalized eigenfunctions at data locations
Psi <- Vh$basis()$eval(as.matrix(locations.obs.2D))

f1 <- as.matrix(Psi %*% f1_normalized)
f2 <- as.matrix(Psi %*% f2_normalized)
f3 <- as.matrix(Psi %*% f3_normalized)

plot_eigenfunction(f1, locations.obs.2D)
plot_eigenfunction(f2, locations.obs.2D)
plot_eigenfunction(f3, locations.obs.2D)

## number of statistical units
M <- 60

## set score standard deviation
sd_score1 <- 0.4
sd_score2 <- 0.3
sd_score3 <- 0.2
sd_error  <- 0.1

seed = 467897965 ## for reproducibility purposes
n_replicas = 30  ## number of repetitions
data_range <- max(c(f1, f2, f3)) - min(c(f1, f2, f3))

#if (!dir.exists("data")) dir.create("data")
#setwd("data")

## export eigenfunctions
write.csv(format(f1, digits = 16), paste(obs_data_path,"f1.csv", sep = "/"))
write.csv(format(f2, digits = 16), paste(obs_data_path,"f2.csv", sep = "/"))
write.csv(format(f3, digits = 16), paste(obs_data_path,"f3.csv", sep = "/"))

for(i in 1:n_replicas) {
    set.seed(seed + 4 * i)
    ## sample scores from normal distribution
    score1 <- rnorm(n = M, sd = sd_score1 * data_range)
    score2 <- rnorm(n = M, sd = sd_score2 * data_range)
    score3 <- rnorm(n = M, sd = sd_score3 * data_range)
    ## generate true observations
    datamatrix_pointwise_exact <-
        matrix(score1) %*% t(matrix(f1)) +
        matrix(score2) %*% t(matrix(f2)) +
        matrix(score3) %*% t(matrix(f3))
    
    ## add error to simulate real data
    error <- rnorm(n = M * n_locations, sd = sd_error * data_range)
    #datamatrix_pointwise <- datamatrix_pointwise_exact + error
    datamatrix_pointwise <- datamatrix_pointwise_exact + matrix(error, #modified sum
                                                                nrow = nrow(datamatrix_pointwise_exact),
                                                                byrow = T)

    ## datamatrix.pointwise are the observations
    ## (pointwise) center data
    data_bar <- colMeans(datamatrix_pointwise)
    data_bar <- matrix(rep(data_bar, nrow(datamatrix_pointwise)),
                       nrow = nrow(datamatrix_pointwise),
                       byrow = TRUE)

    datamatrix_pointwise_centred <- datamatrix_pointwise - data_bar

    ## export data .csv format
    write.csv(format(score1, digits = 16), paste(obs_data_path,paste("score1", "-", i, ".csv", sep = ""),sep="/"))
    write.csv(format(score2, digits = 16), paste(obs_data_path,paste("score2", "-", i, ".csv", sep = ""),sep="/"))
    write.csv(format(score3, digits = 16), paste(obs_data_path,paste("score3", "-", i, ".csv", sep = ""),sep="/"))
    write.csv(format(datamatrix_pointwise_centred, digits = 16),
              paste(obs_data_path,paste("datamatrix_centred", "-", i, ".csv", sep = ""),sep="/")
              )
}

system("/Users/marcogalliani/Desktop/fdaPDE-cpp/cmake-build-default/experiments/fpca/fpca_test --gtest_filter=* --gtest_color=no")

# plot estimated PCs
library(Matrix)
library(gridExtra)

est_PC1<- as.matrix(readMM("est_PC1.mtx"))
est_PC2 <- as.matrix(readMM("est_PC2.mtx"))
est_PC3 <- as.matrix(readMM("est_PC3.mtx"))

grid.arrange(plot_eigenfunction(f1, locations.obs.2D), 
             plot_eigenfunction(-est_PC1, locations.obs.2D),
             plot_eigenfunction(f2, locations.obs.2D), 
             plot_eigenfunction(-est_PC2, locations.obs.2D),
             plot_eigenfunction(f3, locations.obs.2D), 
             plot_eigenfunction(-est_PC3, locations.obs.2D),
             ncol=2, nrow=3)

