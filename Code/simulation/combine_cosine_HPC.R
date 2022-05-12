library(tidyverse)

res = NULL
for (i in 1:200) {
  filename = paste0("../../Data/simu_cosine_task_", i, ".csv")
  # if (!(filename %in% dir())) {next}
  tmp = read_csv(filename, col_types = cols())
  res = rbind(res, tmp)
}
out_name = paste0("../../Data/simulation_cosine.csv")
write_csv(res, out_name)