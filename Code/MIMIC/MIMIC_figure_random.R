library(tidyverse)
source("../../Code/simulation/R_ggplot.R")
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

m_Q1_Q3 = function(m, Q1, Q3, digits=3){
  m = formatC(m, digits, format = "f")
  Q1 = formatC(Q1, digits, format = "f")
  Q3 = formatC(Q3, digits, format = "f")
  return( paste0(m," (", Q1, ", ", Q3, ")") )
}

# table 1 -----------------------------------------------------------------
output = res %>% left_join(disease)
output = rbind(output, output) %>% 
  mutate(case_p = num_pts_last_visit/7519) %>% 
  mutate(group = case_when(row_number()<=400 ~ "1_400",
                           (row_number()>=401 & row_number()<=800) ~ "401_800",
                           (row_number()>=801 & row_number()<=1193) ~ "801_1193",
                           row_number()>=1194 ~ "all")) %>% 
  group_by(group) %>% 
  mutate(m_case_p = median(case_p),
         Q1_case_p = quantile(case_p, 0.25),
         Q3_case_p = quantile(case_p, 0.75),
         
         m_LR = median(LG),
         Q1_LR = quantile(LG, 0.25),
         Q3_LR = quantile(LG, 0.75),
         
         m_RF = median(RF),
         Q1_RF = quantile(RF, 0.25),
         Q3_RF = quantile(RF, 0.75),
         
         m_PV = median(PV_with_time_no_corr),
         Q1_PV = quantile(PV_with_time_no_corr, 0.25),
         Q3_PV = quantile(PV_with_time_no_corr, 0.75),
         
         m_WPV = median(PV_with_time_with_corr),
         Q1_WPV = quantile(PV_with_time_with_corr, 0.25),
         Q3_WPV = quantile(PV_with_time_with_corr, 0.75),
         
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
  distinct() %>% 
  mutate(cases = m_Q1_Q3(m_case_p, Q1_case_p, Q3_case_p),
         LR = m_Q1_Q3(m_LR, Q1_LR, Q3_LR),
         RF = m_Q1_Q3(m_RF, Q1_RF, Q3_RF),
         PV = m_Q1_Q3(m_PV, Q1_PV, Q3_PV),
         WPV = m_Q1_Q3(m_WPV, Q1_WPV, Q3_WPV)) %>% 
  select(group, cases, WPV, PV, LR, RF, n_WPV_PV, n_WPV_LR, n_WPV_RF)

output = as_tibble(t(output), rownames = "var")
# write_csv(output, "table_1.csv")

# plot --------------------------------------------------------------------
D_list = list(c(1,400), c(401,800), c(801,1193))

res = read_csv("../../Data/MIMIC_res.csv")
res = res %>% 
  select(-PV_no_time_no_corr) %>% 
  group_by(ICD) %>% 
  mutate(LG = mean(LG),
         RF = mean(RF_500),
         PV_with_time_no_corr = mean(PV_with_time_no_corr),
         PV_with_time_with_corr = mean(PV_with_time_with_corr)) %>% 
  select(-RF_500) %>% 
  ungroup() %>% 
  distinct() %>% 
  mutate(nn = row_number())

set.seed(1000)
G_all = list()
for (i in 1:length(D_list)) {
  plot_data = res %>% 
    slice(D_list[[i]][1]:D_list[[i]][2]) %>% 
    sample_n(50) %>% 
    gather(key = "method", value = "AUC", LG:RF) %>% 
    mutate(method = case_when(method=="LG"~"regression",
                              method=="RF"~"random forest",
                              method=="PV_with_time_no_corr"~"P2V",
                              method=="PV_with_time_with_corr"~"WeightP2V")) %>% 
    arrange(method, nn)
  
  plot_data$method = factor(plot_data$method,
                            levels = c("WeightP2V", "P2V", "regression", "random forest"))
  plot_data$ICD = factor(plot_data$ICD, levels = plot_data$ICD[1:50])
  
  G_title = expression(paste(LETTERS[i],
                             ". Random 50 diagnoses among top ",
                             underline(D_list[[i]][1], "-", D_list[[i]][2]),
                             " of the most prevalent diagnoses"))
  
  a = as.character(D_list[[i]][1])
  b = as.character(D_list[[i]][2])
  G_title = bquote(bold(.(LETTERS[i])*.~Random~"50"~diagnoses~among~top~
                          underline(bolditalic(.(a)*"-"*.(b)))~
                          of~the~most~prevalent~diagnoses))
  
  temp_theme = my_theme
  temp_theme$legend.position = "top"
  temp_theme$legend.text$size = 16
  temp_theme$axis.text.x$size = 10
  temp_theme$axis.text.x$angle = 45
  temp_theme$axis.text.x$hjust = 1
  G_all[[i]] = ggplot(data = plot_data, aes(x=ICD, y=AUC))+
    geom_point(aes(shape=method, color=method), size=5.5)+
    geom_hline(yintercept=0.5, color="darkgrey", linetype="dashed")+
    scale_y_continuous(limits = c(0.4, 1),
                       breaks = seq(0.4, 1, 0.1))+
    scale_color_manual(values = c("red", "black", "black", "black"))+
    scale_shape_manual(values = 49:52)+
    labs(y="Cross validation average AUC",
         x="Diagnoses",
         title = G_title)+
    temp_theme
  
  g1 = ggplot_build(G_all[[i]])
  G_all[[i]] = G_all[[i]] + 
    scale_shape_manual(values = 49:52,
                       labels = paste("<span style='color:",
                                      c("red", "black", "black", "black"),
                                      "'>",
                                      c("WeightP2V", "P2V", "regression", "random forest"),
                                      "</span>",
                                      sep = ""))+
    scale_color_manual(values = c("red", "black", "black", "black"),
                       labels = paste("<span style='color:",
                                      c("red", "black", "black", "black"),
                                      "'>",
                                      c("WeightP2V", "P2V", "regression", "random forest"),
                                      "</span>",
                                      sep = ""))+
    theme(legend.text = element_markdown())
}

ggsave("MIMIC_figure_random.png",
       plot = plot_grid(plotlist = G_all, ncol = 1),
       width = 20*1, height = 6*3, dpi = 300)




