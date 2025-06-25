library(femR)
library(fdaPDE)

# Mesh ----
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
  # Construct a mesh object in femR
  mesh <- Mesh(
            list(
              nodes = mesh.2D$nodes,
              elements = mesh.2D$triangles,
              boundary = mesh.2D$nodesmarkers
            )
          )
  return(mesh)
}

Psi.matrix <- function(mesh,locations.obs.2D){
  # evaluate the FEM basis at locations
  Vh <- FunctionSpace(mesh, fe_order=1)
  return(Vh$basis()$eval(as.matrix(locations.obs.2D)))
}

unit_square.locs <- function(n, locs_path=NULL){
  ## define observations' locations
  x.obs.2D <- seq(0, 1, length.out = sqrt(n))
  y.obs.2D <- x.obs.2D
  locations.obs.2D <- expand.grid(x.obs.2D, y.obs.2D)
  n_locations <- nrow(locations.obs.2D)
  
  ## export locations
  if(!is.null(locs_path)){
    write.csv(format(locations.obs.2D, digits = 16), paste(locs_path,"locs.csv", sep = "/")) 
  }
  return(locations.obs.2D)
}

# Differential operators----
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

# Eigenfunctions ----
diff_op.eigenfunctions <- function(pde,mesh){
  #' pde: femR pde object
  #' mesh: femR mesh object
  #' 
  evd <- geigen::geigen(A=as.matrix(pde$stiff()),B=as.matrix(pde$mass()))
  return(as.matrix(evd$vectors[,rev(1:ncol(evd$vectors))]))
}

laplacian_square_eigenfunction <- function(a, b, locs, x_t=0,y_t=0) {
  return(cos(a * pi * (locs[,1]-x_t)) * cos(b * pi * (locs[, 2]-y_t)))
}

L2norm <- function(f, R0) { return(sqrt(as.numeric(t(f) %*% R0 %*% f))) }

normalize <- function(f,R0){
  norm <- sqrt(as.numeric(t(f) %*% R0 %*% f))
  return(f/norm)
}

## Plot ---
library(ggplot2)
library(viridis)

plot_eigenfunction <- function(data, locs, title=NULL) {
  ## define data.frame for plotting
  df <- data.frame(
    x = locs[,1],
    y = locs[,2],
    f = data
  )
  ## colors
  n_breaks <- 20
  mybreaks <- c(-Inf, seq(min(data,na.rm=T), max(data,na.rm=T), length.out = n_breaks), Inf)
  mycolors<- function(x) {
    colors<-colorRampPalette(viridis(11))( x + 1 )
    colors[1:x]
  }
  ## plot
  p <- ggplot() +
    geom_contour_filled(data = df, aes(x, y, z = f),
                        breaks = mybreaks, size=0.1) +
    scale_fill_manual(
      aesthetics = "fill",
      values = mycolors(n_breaks + 2), name = "Value", drop = FALSE
    ) +
    coord_equal() + 
    theme_void() +
    theme(legend.position = "none",plot.title = element_text(hjust = 0.5))
  if(!is.null(title)){
    p <- p + ggtitle(title)
  }
  
  p
}

plot_stat_units <- function(stat_unit, locations, title=NULL, mybreaks=NULL, isolines=FALSE){
  ## data
  df <- data.frame(
    x = locations[,1],
    y = locations[,2],
    f = stat_unit
  )
  ## colors
  n_breaks <- 50
  if(is.null(mybreaks)){
    mybreaks <- c(-Inf, seq(min(df$f,na.rm=T), max(df$f,na.rm=T),
                            length.out = n_breaks), 
                  Inf)
  }
  mycolors <- colorRampPalette(viridis(11))(length(mybreaks) - 1)
  
  x_range <- range(locations[,1], na.rm = TRUE)
  y_range <- range(locations[,2], na.rm = TRUE)
  
  plot <- ggplot(df, aes(x = x, y = y, fill = f)) +
    geom_tile() +
    scale_fill_gradientn(
      colors = mycolors,
      breaks = mybreaks,
      name = "Value",
      na.value = "white"
    ) +
    coord_fixed(xlim = x_range, ylim = y_range, expand=F) +
    theme_void() +
    theme(legend.position = "none",
          #plot.title = element_text(hjust = 0.8),
          panel.border = element_rect(color = "black", fill = NA, size = 0.5),
          text = element_text(size=20))
  
  if(!is.null(title)){
    plot <- plot + ggtitle(title)
  }
  
  if(isolines){
    plot <- plot + geom_contour(data=df, aes(x = x, y = y, z = f), 
                                color = "black", breaks = mybreaks, size=0.2)
  }
  return(plot)
  
}

