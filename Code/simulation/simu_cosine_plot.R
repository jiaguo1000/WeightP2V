library(tidyverse)
source("R_ggplot.R")

# plot --------------------------------------------------------------------
rawdata = read_csv("../../Data/simulation_cosine.csv")
l_list = c("signal", "noise")
t_list = c(
  expression(atop("A. Cosine similarities between the vector of the outcome concept and"~
                    "vectors of"~underline(bolditalic("signal"))~"concepts")),
  expression(atop("B. Cosine similarities between the vector of the outcome concept and"~
                    "vectors of"~underline(bolditalic("noise"))~"concepts"))
)

G = list()
tmp_theme = my_theme
for (i in 1:length(l_list)) {
  tt = t_list[i]
  myd = rawdata %>% 
    filter(group==l_list[i]) %>% 
    group_by(beta) %>% 
    summarise(corr_m = median(corr),
              qL = quantile(corr, 0.25),
              qH = quantile(corr, 0.75))
  
  tmp_theme$plot.title$size = 17
  tmp_theme$axis.title$size = 17
  tmp_theme$legend.key.width = unit(2.3, "lines")
  G[[i]] = ggplot(data = myd, aes(x=beta, y=corr_m))+
    geom_line(size=1.2)+
    geom_point(size=1.5)+
    geom_errorbar(aes(ymin=qL, ymax=qH), size=0.8, width=0.05, alpha=0.4)+
    scale_y_continuous(limits = c(-1, 1))+
    scale_x_continuous(breaks = seq(-1.0, 1.0, 0.2), labels = seq(-1.0, 1.0, 0.2))+
    labs(y="Cosine similarity",
         x=expression(paste(beta, " of signal features")),
         title = tt)+
    tmp_theme
}

ggsave("figure.png",
       plot = plot_grid(plotlist = G, ncol = 1, nrow = 2),
       width = 11.5*1, height = 5*2, dpi = 300)


