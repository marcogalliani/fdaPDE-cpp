library(ggplot2)
library(scales)
library(magrittr)
library(dplyr)
library(viridis)

grouped_boxplot <- function(test_results, 
                            var_name.x,var_name.y,
                            group_name, color_map,
                            title = NULL,
                            limits.y=NULL, log.y=F,
                            lines=F, legend=F, 
                            text.size = 30){
  plot <- ggplot(test_results, aes(x=get(var_name.x), y=get(var_name.y), fill=get(group_name))) + 
    geom_boxplot() +
    scale_fill_manual(values = color_map) +
    theme_light() +
    labs(x=var_name.x, y=var_name.y,fill=group_name)  +
    theme(text = element_text(size=text.size))
  
  if(!is.null(title)){
    plot <- plot + ggtitle(title)
  }
    
  #options
  if(lines){
    plot <- plot +
      stat_summary(
        fun = median,
        geom = 'line',
        aes(group = get(group_name), color = get(group_name)),
        position = position_dodge(width = 0.85) 
      ) +
      scale_color_manual(values = color_map)
  }
  #log scale
  if(log.y){
    plot <- plot + scale_y_log10(limits=limits.y)
  }
  #legend
  if(!legend){
    plot <- plot + guides(color="none",fill="none")  
  }
  return(plot)
}


grouped_lineplot <- function(test_results, 
                             var_name.x,var_name.y,
                             group_name, color_map,
                             title = NULL,
                             log.x=F,log.y=F,
                             complexity.lines=NULL, legend = F){
  test_results.grouped <- test_results %>% 
    group_by(group=get(group_name), x=get(var_name.x)) %>%
    summarize(y = median(get(var_name.y))) %>%
    mutate(x = as.numeric(as.character(x)))
  
  plot <- ggplot(test_results.grouped, 
                 aes(x=x, y=y, color=group)) + 
    geom_line() +
    theme_light() +
    labs(x=var_name.x, y=var_name.y) +
    scale_color_manual(values=color_map) +
    theme(text = element_text(size=25))
  
  if(!is.null(title)){
    plot <- plot + ggtitle(title)
  }
  
  if(!legend){
    plot <- plot + guides(color="none")  
  }
  
  if(log.x){
    plot <- plot + scale_x_log10()
  }
  if(log.y){
    plot <- plot + scale_y_log10() 
  }
  if(log.x & log.y & !is.null(complexity.lines)){
    min.x <- min(test_results.grouped$x)
    min.y <- min(test_results.grouped$y)
    for(line in complexity.lines){
      plot <- plot + geom_abline(intercept = log10(min.y/min.x^line), slope = line,
                                 linetype = "dotted", color = "grey", linewidth = 0.3)
      label_x <- 0.8 * max(test_results.grouped$x)  # Place label near the right edge of the plot
      label_y <- 10^(log10(min.y/min.x^line) + line * log10(label_x))
      
      plot <- plot + annotate("text", 
                              x = label_x, 
                              y = label_y, 
                              label = paste(var_name.x, line, sep = "^"), 
                              hjust = 0.5, color = "grey", size = 5)
    }
  }
  return(plot)
}

stacked_barplot <- function(test_results, 
                            var_name.part1, var_name.part2,
                            var_name.x,
                            group_name,
                            title){
  
  test_results.extended <- tidyr::pivot_longer(test_results, cols = c(var_name.part1, var_name.part2),
                                  names_to = "type", values_to = "composition")
  
  ggplot(test_results.extended, aes(x = get(var_name.x), y = composition, fill = type)) +
    geom_bar(stat = "identity", position = "fill") +  # 'fill' makes it a percentage plot
    scale_y_continuous(labels = scales::percent) +    # Show percentages on y-axis
    labs(
      x = var_name.x,
      y = paste("%",var_name.part1,sep=""),
      fill = "composition",
      title = title
    ) +
    facet_wrap(~ get(group_name), scales = "free_x") +
    theme_light() + theme(legend.position = "none") +
    theme(text = element_text(size=20))
}



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

