library(tidyverse)
source("R_ggplot.R")

# plot --------------------------------------------------------------------
rawdata = read_csv("../../Data/simulation_prediction.csv", col_types = cols())

p_list = c(0.5, 0.3, 0.1, 0.05)
t_list = c("1:1", "3:7", "1:9", "1:19")
G = list()
k = 1
for (p in p_list) {
  res = rawdata %>% filter(case_p==p)
  qm = res %>% 
    group_by(beta) %>% 
    summarise_all(mean) %>% 
    gather(key = "method", value = "AUC", LG:PV_2) %>% 
    filter(method %in% c("LG", "RF_500", "PV_0", "PV_2")) %>% 
    select(beta, method, AUC)
  
  ql = res %>% 
    group_by(beta) %>% 
    summarise_all(function(x){quantile(x, 0.25)}) %>% 
    gather(key = "method", value = "AUC_L", LG:PV_2) %>% 
    filter(method %in% c("LG", "RF_500", "PV_0", "PV_2")) %>% 
    select(beta, method, AUC_L)
  
  qh = res %>% 
    group_by(beta) %>% 
    summarise_all(function(x){quantile(x, 0.75)}) %>% 
    gather(key = "method", value = "AUC_H", LG:PV_2) %>% 
    filter(method %in% c("LG", "RF_500", "PV_0", "PV_2")) %>% 
    select(beta, method, AUC_H)
  
  plot_data = qm %>% 
    left_join(ql) %>% 
    left_join(qh) %>% 
    mutate(method = case_when(method=="LG"~"regression", method=="RF_500"~"random forest",
                              method=="PV_0"~"P2V", method=="PV_2"~"WeightP2V")) %>% 
    mutate(method = factor(method, levels = c("WeightP2V", "P2V", "regression", "random forest")))
  
  temp_theme = my_theme
  temp_theme$legend.position = "none"
  temp_theme$plot.title$face = "bold"
  
  G[[k]] = ggplot(data = plot_data, aes(x = beta, y = AUC))+
    geom_line(aes(linetype=method), size=1.8)+
    geom_point(size=1.5)+
    geom_errorbar(aes(ymin=AUC_L, ymax=AUC_H), size=0.8, width=0.02, alpha=0.5)+
    scale_x_continuous(breaks = seq(0.0, 1.0, 0.2), labels = seq(0.0, 1.0, 0.2))+
    scale_y_continuous(limits = c(0.35, 1.0), breaks = seq(0.4, 1.0, 0.1))+
    # scale_color_manual(values = my_color(4))+
    scale_linetype_manual(values = c("solid", "dashed", "dotdash", "dotted"))+
    labs(title = paste0(LETTERS[k], ". case:control=", t_list[k]),
         x = expression(beta))+
    temp_theme
  k = k+1
}
main_plot = plot_grid(plotlist = G, ncol = 2)

temp_theme = my_theme
temp_theme$legend.position = "top"
temp_theme$legend.key.width = unit(4.5, "lines")
legend_plot = get_legend(ggplot(data = plot_data, aes(x = beta, y = AUC))+
                           geom_line(aes(linetype=method), size=1.8)+
                           geom_point(size=1.5)+
                           # scale_color_manual(values = my_color(4))+
                           scale_linetype_manual(values = c("solid", "dashed", "dotdash", "dotted"))+
                           temp_theme)

G_title = ggdraw() + 
  draw_label(paste0("Simulation scenarios under different case/control ratios"),
             fontface = "bold", size = 22)

G_out = plot_grid(G_title, legend_plot, main_plot,
                  ncol = 1, rel_heights = c(0.06, 0.05, 1))

ggsave("simu_AUC.png",
       plot = G_out,
       width = 7*2, height = 6*2, dpi = 300)


