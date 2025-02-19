st_components <- function(n_nodes, t_steps,data_generator_path){
  
  source("../r-utils/mesh_utils.R")
  #->spatio temporal mesh
  mesh.2D <- unit_square.mesh(n_nodes)
  time.grid <- seq(0,1,length.out=t_steps)
  
  
  mesh.st <- expand.grid(seq(0,1,length.out=sqrt(n_nodes)),
                         seq(0,1,length.out=sqrt(n_nodes)),
                         seq(0,1,length.out=t_steps))
  
  ##->3D cube mesh
  mesh_path <- "../data/mesh/unit_cube"
  
  nodes.grid <- expand.grid(seq(0,1,length.out=sqrt(n_nodes)),
                            seq(0,1,length.out=sqrt(n_nodes)),
                            seq(0,1,length.out=t_steps))
  mesh.3D <- fdaPDE::create.mesh.3D(nodes.grid,
                                    build_tetrahedrons(sqrt(n_nodes),sqrt(n_nodes),t_steps))
  
  write.csv(format(mesh.3D$nodes, digits = 16), 
            paste(mesh_path,"points.csv", sep = "/"))
  write.csv(format(mesh.3D$tetrahedrons, digits = 16), 
            paste(mesh_path,"elements.csv", sep = "/"))
  write.csv(format(1 * mesh.3D$nodesmarkers, digits = 16), 
            paste(mesh_path,"boundary.csv", sep = "/"))
  write.csv(format(mesh.3D$neighbors, digits = 16), 
            paste(mesh_path,"neigh.csv", sep = "/"))
  write.csv(format(mesh.3D$edges, digits = 16), 
            paste(mesh_path,"edges.csv", sep = "/"))
  
  
  ##->decompose R1
  library(jsonlite)
  
  path_to_json <- paste(data_generator_path,"params.json",sep="/")
  json_data <- fromJSON(path_to_json)
  
  json_data$RunParams$EVDType <- unbox("exact")
  json_data$RunParams$n_eigenvectors <- unbox(30)
  print("good")
  json_data$DiffusionOp$alpha1 <- unbox(1/3*pi)
  json_data$DiffusionOp$alpha2 <- unbox(4/3*pi)
  json_data$DiffusionOp$gamma1 <- unbox(2)
  json_data$DiffusionOp$gamma2 <- unbox(1.5)
  
  write_json(json_data, path_to_json, pretty = TRUE)
  system(paste("cd ",data_generator_path,"; ","/Users/marcogalliani/Desktop/fdaPDE-cpp/cmake-build-default/experiments/spatio-temporal-data/st-data-generation"))
  
  results_path <- paste(data_generator_path,"results",sep="/")
  R0 <- as.matrix(readMM(file=paste(results_path,"R0.mtx",sep="/")))
  R1.eigenvectors <- as.matrix(readMM(file=paste(results_path,"eigenvectorR1.mtx",sep="/")))
  L2norm <- function(f){
    return(sqrt(as.numeric(t(f) %*% R0 %*% f)))
  }
  
  i1 <- 5
  i2 <- 10
  i3 <- 15
  
  f1 <- matrix(R1.eigenvectors[,i1]/L2norm(R1.eigenvectors[,i1]), ncol=t_steps)
  f2 <- matrix(R1.eigenvectors[,i2]/L2norm(R1.eigenvectors[,i2]), ncol=t_steps)
  f3 <- matrix(R1.eigenvectors[,i3]/L2norm(R1.eigenvectors[,i3]), ncol=t_steps)
  
  return(cbind(c(f1),c(f2),c(f3)))
}

build_tetrahedrons <- function(n.x,n.y,n.z){
  
  tetrahedrons <- matrix(0, nrow = 5*(n.x-1)*(n.y-1)*(n.z-1), ncol = 4)
  j <- 1
  for (z in seq_len(n.z - 1)) {
    for (y in seq_len(n.y - 1)) {
      for (x in seq_len(n.x - 1)) {
        ## build vector of vertices of i-th subcube
        p <- x + (y - 1) * n.y + (z - 1) * n.z^2 ## base point
        v <- c(p, p + 1, p + n.y, p + n.y + 1, p + n.z^2, p + n.z^2 + 1, p + n.z^2 + n.y, p + n.z^2 + n.y + 1)
        ## compute vertices of each thetraedron in the subcube
        tetrahedrons[j,     ] <- c(v[1], v[2], v[3], v[5])
        tetrahedrons[j + 1, ] <- c(v[2], v[3], v[4], v[8])
        tetrahedrons[j + 2, ] <- c(v[2], v[3], v[5], v[8])
        tetrahedrons[j + 3, ] <- c(v[2], v[5], v[6], v[8])
        tetrahedrons[j + 4, ] <- c(v[3], v[5], v[7], v[8])
        j <- j + 5
      }
    }
  }
  return(tetrahedrons)
}