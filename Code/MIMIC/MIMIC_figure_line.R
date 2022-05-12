library(tidyverse)
source("../../Code/simulation/R_ggplot.R")
res = read_csv("../../Data/MIMIC_res.csv")

disease = read_csv("../../Data/MIMIC_outcome_prevalence.csv") %>% 
  select(-if_last_then_num_pts_prev_visits, -if_last_then_num_seq_prev_visits)

# function ----------------------------------------------------------------
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

n = nrow(res)
identical(res$ICD, disease$ICD[1:n])

m_Q1_Q3 = function(m, Q1, Q3, digits=3){
  m = formatC(m, digits, format = "f")
  Q1 = formatC(Q1, digits, format = "f")
  Q3 = formatC(Q3, digits, format = "f")
  return( paste0(m," (", Q1, ", ", Q3, ")") )
}

# group -------------------------------------------------------------------
output = res %>% left_join(disease) %>% mutate(case_p = num_pts_last_visit/7519)
output = rbind(output, output)
output$group = NA

n = nrow(res)
k = 200
cut_list = floor(n/k)
cut_list = (0:cut_list)*k
cut_list = c(cut_list, n)
for (i in 1:(length(cut_list)-1)) {
  lhs = cut_list[i]+1
  rhs = cut_list[i+1]
  tmp = paste0(lhs, "_", rhs)
  output = output %>% 
    mutate(group = ifelse(row_number()>=lhs & row_number()<=rhs, tmp, group))
}
output = output %>% 
  mutate(group = ifelse(row_number()>=(n+1), "all", group))

l = 0.25
h = 0.75
output = output %>% 
  group_by(group) %>% 
  mutate(m_case_p = median(case_p),
         Q1_case_p = quantile(case_p, l),
         Q3_case_p = quantile(case_p, h),
         
         m_LR = median(LG),
         Q1_LR = quantile(LG, l),
         Q3_LR = quantile(LG, h),
         
         m_RF = median(RF),
         Q1_RF = quantile(RF, l),
         Q3_RF = quantile(RF, h),
         
         m_PV = median(PV_with_time_no_corr),
         Q1_PV = quantile(PV_with_time_no_corr, l),
         Q3_PV = quantile(PV_with_time_no_corr, h),
         
         m_WPV = median(PV_with_time_with_corr),
         Q1_WPV = quantile(PV_with_time_with_corr, l),
         Q3_WPV = quantile(PV_with_time_with_corr, h),
         
         n_WPV_PV = sum(PV_with_time_with_corr>PV_with_time_no_corr),
         n_WPV_LR = sum(PV_with_time_with_corr>LG),
         n_WPV_RF = sum(PV_with_time_with_corr>RF)) %>% 
  ungroup() %>% 
  select(group, m_case_p, Q1_case_p, Q3_case_p,
         m_LR, Q1_LR, Q3_LR,
         m_RF, Q1_RF, Q3_RF,
         m_PV, Q1_PV, Q3_PV,
         m_WPV,Q1_WPV, Q3_WPV,
         n_WPV_PV, n_WPV_LR, n_WPV_RF) %>% 
  distinct()

# plot --------------------------------------------------------------------
output = output[output$group!="all",]
plot_data = tibble(rank = rep(1:length(output$m_LR), 4),
                   AUC = c(output$m_LR, output$m_RF, output$m_PV, output$m_WPV),
                   lower = c(output$Q1_LR, output$Q1_RF, output$Q1_PV, output$Q1_WPV),
                   upper = c(output$Q3_LR, output$Q3_RF, output$Q3_PV, output$Q3_WPV),
                   method = rep(c("regression", "random forest", "P2V", "WeightP2V"),
                                each=length(output$m_LR)))

x_lab = formatC(output$m_case_p, 3, format = "f")
x_lab = paste0("Rank: ", str_replace(output$group, "_", "-"),
               "\n median p=", x_lab)

plot_data$method = factor(plot_data$method,
                          levels = c("WeightP2V", "P2V", "regression", "random forest"))

temp_theme = my_theme
temp_theme$axis.title.x = element_blank()
temp_theme$axis.text.x$size = 11
temp_theme$legend.position = "right"
temp_theme$legend.key.width = unit(4, "lines")
G = ggplot(data = plot_data, aes(x=rank, y=AUC))+
  geom_line(aes(linetype=method), size=1.5)+
  geom_point(size=1.5)+
  geom_errorbar(aes(ymin=lower, ymax=upper), size=0.8, width=0.08, alpha=0.6)+
  scale_x_continuous(breaks = 1:length(output$m_LR), labels = x_lab)+
  scale_y_continuous(limits = c(0.3, 1.0), breaks = seq(0.3, 1.0, 0.1))+
  scale_color_manual(values = my_color(4))+
  scale_linetype_manual(values = c("solid", "dashed", "dotdash", "dotted"))+
  temp_theme

ggsave("MIMIC_figure_line.png",
       plot = G,
       width = 11, height = 7, dpi = 300)






