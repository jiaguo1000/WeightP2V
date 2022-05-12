library(tidyverse)

res = read_csv("../../Data/MIMIC_res.csv")

disease = read_csv("../../Data/MIMIC_outcome_prevalence.csv") %>% 
  select(-if_last_then_num_pts_prev_visits, -if_last_then_num_seq_prev_visits)

res = res %>% 
  select(-PV_no_time_no_corr) %>% 
  group_by(ICD) %>% 
  mutate(LG = mean(LG),
         RF = mean(RF_500),
         PV_with_time_no_corr = mean(PV_with_time_no_corr),
         PV_with_time_with_corr = mean(PV_with_time_with_corr)) %>% 
  select(-RF_500) %>% 
  ungroup() %>% 
  distinct()

output = res %>% 
  left_join(disease) %>% 
  rename(AUC_LR = LG,
         AUC_RF = RF,
         AUC_P2V = PV_with_time_no_corr,
         AUC_WeightP2V = PV_with_time_with_corr) %>% 
  mutate(prevalence = num_pts_last_visit/7519) %>% 
  select(ICD, SHORT_TITLE, LONG_TITLE, prevalence,
         AUC_WeightP2V, AUC_P2V, AUC_LR, AUC_RF)

write_csv(output, "../../Data/Supplement_B.csv")
