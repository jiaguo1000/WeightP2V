library(tidyverse)

res = NULL
for (i in 1:1193) {
  filename = paste0("../../Data/MIMIC_task_", i, ".csv")
  tmp = read_csv(filename)
  res = rbind(res, tmp)
}
out_name = paste0("../../Data/MIMIC_res.csv")
write_csv(res, out_name)