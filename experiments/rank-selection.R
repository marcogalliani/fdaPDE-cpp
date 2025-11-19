model_name <- "fpca-na-gcv"
cpp_path <- "fpca-na-gcv"

library(jsonlite)
json_file <- jsonlite::read_json(paste0(cpp_path,"/params.json"))

cv_scores <- list()
for(i in 1:5){
  cat("Rank: ",i,"\n")
  json_file$RunParams$n_pc <- i
  jsonlite::write_json(json_file,paste0(cpp_path,"/params.json"),pretty=T,auto_unbox=T)
  system(paste0("cd ", model_name, " && ", "./fit_model"),
         ignore.stdout = F)
  cv_scores[[i]] <- as.matrix(read.csv(paste0(model_name,"/test-results/gcv_scores.csv")))
}
cv_scores_mat <- do.call(cbind, cv_scores)
cv_scores_mat


matplot(cv_scores_mat,type="l",log="y")
