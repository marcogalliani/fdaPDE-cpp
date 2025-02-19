library(fdaPDE)
library(femR)

# Mesh Type----
unit_square.mesh <- function(N, mesh_path=NULL){
  # N: number of nodes in the mesh

  x.2D <- seq(0, 1, length.out = sqrt(N))
  y.2D <- x.2D
  locations.2D <- expand.grid(x.2D, y.2D)
  mesh.2D <- fdaPDE::create.mesh.2D(locations.2D)
  
  ## export mesh
  if(!is.null(mesh_path)){
    write.csv(format(mesh.2D$nodes, digits = 16), paste(mesh_path,"points.csv", sep = "/"))
    write.csv(format(mesh.2D$triangles, digits = 16), paste(mesh_path,"elements.csv", sep = "/"))
    write.csv(format(1 * mesh.2D$nodesmarkers, digits = 16), paste(mesh_path,"boundary.csv", sep = "/"))
    write.csv(format(mesh.2D$neighbors, digits = 16), paste(mesh_path,"neigh.csv", sep = "/"))
    write.csv(format(mesh.2D$edges, digits = 16), paste(mesh_path,"edges.csv", sep = "/"))
  }
  return(mesh.2D)
}

unit_square.locs <- function(n, locs_path=NULL){
  ## define observations' locations
  x.obs.2D <- seq(0, 1, length.out = sqrt(n))
  y.obs.2D <- x.obs.2D
  locations.obs.2D <- expand.grid(x.obs.2D, y.obs.2D)
  n_locations <- nrow(locations.obs.2D)

  ## export locations
  if(!is.null(locs_path)){
    write.csv(format(locations.obs.2D, digits = 16), paste(obs_data_path,"locs.csv", sep = "/")) 
  }
  return(locations.obs.2D)
}

# EigenFunctions----
square_eigenfunction <- function(a, b, locs) {
  return(cos(a * pi * locs[,1]) * cos(b * pi * locs[, 2]))
}

anisothropic_diffusion_op <- function(alpha, gamma){
  R <- matrix(
    c(cos(alpha),-sin(alpha),
      sin(alpha),cos(alpha)),
    nrow=2, ncol=2
  )
  Sigma <- matrix(
    c(1/sqrt(gamma),0,
      0,sqrt(gamma)),
    nrow=2, ncol=2
  )
  #define K
  K <- R %*% Sigma %*% t(R)
  
  return(
    function(f){ 
      return(femR::div(K*femR::grad(f)))
    }
  )
}

eigenfunctions.Lop <- function(n_pc,L,u,f,mesh.2D,Psi){
  L <- anisothropic_diffusion_op(
    pde[[diff]]$diff_params[1],
    pde[[diff]]$diff_params[2]
  )
  #L2 norm
  L2norm <- L2norm.factory(mesh.2D, L) #function to compute the norm
  #evd
  evd <- eigen(Pde(L(u),f)$stiff(), symmetric = T, only.values = F)
  
  #eigen functions
  f1_fem <- evd$vectors[,2]
  f2_fem <- evd$vectors[,3]
  f3_fem <- evd$vectors[,4]
  
  f1 <- as.matrix(Psi %*% f1_fem/L2norm(f1_fem))
  f2 <- as.matrix(Psi %*% f2_fem/L2norm(f2_fem))
  f3 <- as.matrix(Psi %*% f3_fem/L2norm(f3_fem))
  
  return(cbind(f1,f2,f3))
}

L2norm.factory <- function(mesh.2D, diff_op=function(u){return(-laplace(u))}){
  #define a dummy pde to get the mass matrix
  mesh <- Mesh(
    list(
      nodes = mesh.2D$nodes,
      elements = mesh.2D$triangles,
      boundary = mesh.2D$nodesmarkers
    )
  )
  Vh <- FunctionSpace(mesh, fe_order=1)
  u <- Function(Vh)
  Lu <- diff_op(u) 
  f <- function(points) { return(0*points[,1])}
  pde <- Pde(Lu, f)
  
  R0 <- pde$mass() ## mass matrix
  
  L2norm <- function(f) { return(sqrt(as.numeric(t(f) %*% R0 %*% f))) }
  return(L2norm)
}

Psi.matrix <- function(mesh.2D,locations.obs.2D){
  mesh <- Mesh(
    list(
      nodes = mesh.2D$nodes,
      elements = mesh.2D$triangles,
      boundary = mesh.2D$nodesmarkers
    )
  )
  ## define a dummy PDE, just to get the mass matrix
  Vh <- FunctionSpace(mesh, fe_order=1)
  
  return(Vh$basis()$eval(as.matrix(locations.obs.2D)))
}

# Data generation----
data_generator <- function(M, true_PCs,scores_sd,error_sd,seed=42,obs_path=NULL){
  set.seed(seed)
  data <- list()
  data_range <- max(true_PCs) - min(true_PCs)
  data$rank <- ncol(true_PCs)
  data$pcs <- true_PCs
  
  #(1) scores & exact data matrix
  data$scores <- matrix(nrow=M, ncol=ncol(true_PCs))
  for(i in 1:ncol(true_PCs)){
    data$scores[,i] <- rnorm(n = M, sd = scores_sd[i] * data_range)
    data$scores[,i] <- data$scores[,i] - mean(data$scores[,i])
  }
  #(2) compute the noisy data matrix
  error <- rnorm(n = M*nrow(true_PCs), sd = error_sd*data_range)
  error <- matrix(error,nrow=M)
  error <- error - matrix(rep(colMeans(error),M), byrow=T, nrow=M)
  
  data$noisy_data <- data$scores %*% t(data$pcs) + error
  
  #(3) centering
  ## => guaranteed by centering scores and errors
  
  #(4) save the noisy matrix
  if(!is.null(obs_path)){
    write.csv(format(data$noisy_data, digits = 16),
              paste(obs_path,"datamatrix_centred.csv",sep="/"))
  }
  return(data)
}

# copied from: https://github.com/fdaPDE/case-studies/blob/main/lake_victoria/censoring.R
createNA<-function(Data, p=0.75, schema="a", mesh_ref=NULL, locations=NULL, RDD_groups=NULL, mesh_ref_time=c(0,1), timelocations=NULL, w=8)
{
  if(schema=="a") # ind in space, ind in time
  {
    data_vector=as.vector(Data)
    size=length(data_vector)
    data_vector[sample(1:size,size-round(size*p))]=NA
    Data=matrix(data = data_vector,nrow = nrow(Data),ncol = ncol(Data))
  }
  if(schema=="b") # ind in space, dep in time
  {
    nodes=mesh_ref_time
    nobs=RDD_groups*p
    pts=timelocations
    for(s in 1:nrow(Data))
    {
      Itime = sample(1:length(mesh_ref_time),RDD_groups)
      seeds= nodes[Itime]
      Iobs=sample(1:RDD_groups,nobs)
      Obsmarker=rep(0,RDD_groups)
      Obsmarker[Iobs]=1
      for(k in 1:length(pts))
      {
        if(Obsmarker[nearest(c(pts[k],0),cbind(seeds,rep(0,length(seeds))))]==0)
        {
          Data[s,k]=NA
        }
      }
    }
  }
  if(schema=="c") # dep in space, ind in time
  {
    nodes=mesh_ref$nodes
    nobs=RDD_groups*p
    pts=locations
    
    for(t in 1:ncol(Data))
    {
      Ispace = sample(1:nrow(nodes),RDD_groups)
      seeds= nodes[Ispace,]
      Iobs=sample(1:RDD_groups,nobs)
      Obsmarker=rep(0,RDD_groups)
      Obsmarker[Iobs]=1
      
      for(k in 1:nrow(pts))
      {
        if(Obsmarker[nearest(pts[k,],seeds)]==0)
        {
          Data[k,t]=NA
        }
      }
    }
  }
  if(schema=="d") # dep in space, dep in time
  {
    Datavec=as.vector(Data)
    nodes=mesh_ref$nodes
    Ispace = sample(1:nrow(nodes),RDD_groups)
    seedstime=runif(RDD_groups,head(mesh_ref_time,1),tail(mesh_ref_time,1))
    seeds= cbind(nodes[Ispace,1],nodes[Ispace,2],seedstime)
    nobs=RDD_groups*p
    Obsmarker=rep(0,RDD_groups)
    Iobs=sample(1:RDD_groups,nobs)
    Obsmarker[Iobs]=1
    pts=cbind(rep(locations[,1],length(timelocations)),rep(locations[,2],length(timelocations)),rep(timelocations,each=nrow(locations)))
    for(k in 1:nrow(pts))
    {
      if(Obsmarker[nearest(pts[k,],seeds,w)]==0)
      {
        Datavec[k]=NA
      }
    }
    Data=matrix(data = Datavec, nrow = nrow(locations),ncol=length(timelocations))
    
  }
  Data
}

