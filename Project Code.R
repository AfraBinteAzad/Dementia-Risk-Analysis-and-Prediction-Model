install.packages('tidyverse')
installed.packages('Amelia')
install.packages('readxl')
library(tidyverse)
library(Amelia)
library(readxl)

df <- read_excel("C:/Users/User/Desktop/Dementia Risk Prediction Model/dataset.xlsx")
head(df)

colnames(df)
df

#EDA
categorical <- c("Education_ID","Mobility","MNAa_q3","Hyperlipidaemia","MMSE_class")
continous <- c("Age","body_height","body_weight","MNAb_tot","waist","MNAa_tot")

df %>% select(all_of( categorical)) %>%  map(table)
df %>% select(all_of( continous)) %>%
  map(., ~{
    c(
      Mean = mean(.x, na.rm = T) %>% round(2),
      SD = sd(.x, na.rm = T)%>% round(2)
    )
  }
  )

#Target variable distribution

target_distribution <- 
  ggplot(df, aes(x = MMSE_class, fill = factor(MMSE_class))) +
  geom_bar() +
  ggtitle("Distribution of MMSE Class")

target_distribution

#Null Value Handling:

colSums(is.na(df))
round(colMeans(is.na(df))*100,2)
total_missingness <- sum(is.na(df))/(nrow(df)*ncol(df))*100
total_missingness

missing_df <- data.frame(
  Variable = names(df),
  MissingPercent = round(colMeans(is.na(df)) * 100, 2)
)
ggplot(missing_df, aes(x = reorder(Variable, -MissingPercent), 
                       y = MissingPercent)) +
  geom_col(fill = "steelblue") +
  labs(title = "Percentage of Missing Values per Variable",
       x = "Variable",
       y = "Missing (%)")
missmap(df,main="Missing Map",col=c('yellow','black'),Legend=FALSE)

#data imputation
df <- df %>% filter(!is.na(MMSE_class))
round(colMeans(is.na(df))*100,2)
for(i in continous){
  df[[i]][is.na(df[[i]])] <- median(df[[i]],na.rm=TRUE)
}

get_mode <- function(v){
  uniq <- na.omit(unique(v))
  uniq[which.max(tabulate(match(v,uniq)))]
}

for(i in categorical){
  df[[i]][is.na(df[[i]])] <- get_mode(df[[i]])
}
round(colMeans(is.na(df))*100,2)
missmap(df,main="Missing Map",col=c('yellow','black'),Legend=FALSE)

#data factorization
df <- df %>% mutate(
  Education_ID=factor(Education_ID,
                      levels = c(1,2,3,4),
                      labels = c('No formal education','Elementary school level','High school level','College level or above')),
  Mobility=factor(Mobility,
                  levels = c(1,2,3,4),
                  labels = c('Wheelchair','Walker','Cane','Self-mobility')
  ),
  MNAa_q3=factor(MNAa_q3,
                 levels = c(0,1,2),
                 labels = c('Bedridden','Cannot go out','Independent')
  ),
  Hyperlipidaemia=factor(Hyperlipidaemia,
                         levels = c(0,1),
                         labels = c('No','Yes')
  ),
  MMSE_class=factor(MMSE_class,
                    levels = c(0,1),
                    labels = c('Not at risk','At risk')
  )
)


#feature engineering
df <- df %>% mutate(
  BMI= body_weight/(body_height/100)^2
)
df$BMI

str(df) 
summary(df)
