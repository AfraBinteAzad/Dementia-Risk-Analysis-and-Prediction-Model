install.packages('tidyverse')
installed.packages('Amelia')
install.packages('readxl')
library(tidyverse)
library(Amelia)
library(readxl)

df <- read_excel("C:/Users/User/Desktop/Dementia Risk Prediction Model/dataset.xlsx")
head(df)

colnames(df)



#EDA
categorical <- c("Education_ID","Mobility","MNAa_q3","Hyperlipidemia","MMSE_class")
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


#categorical value distribution graph

cat_plot <- function(data, category) {
  data %>%
    ggplot(aes(
      x = factor(.data[[category]]),
      fill = factor(MMSE_class)
    )) +
    geom_bar() +
    xlab(category) +
    ggtitle(paste("Distribution of", category))
}

all_cat_plots <- map(categorical, ~ cat_plot(df, .x))
all_cat_plots


#continous value distribution graph

cont_plot <- function(data, conti) {
  data %>%
    ggplot(
      aes(
        x = .data[[conti]],
        fill = factor(MMSE_class)
      )
    ) +
    geom_histogram(bins = 30,position = "identity") +
    labs(
      title = paste("Distribution of", conti),
      x = conti,
      y = "Frequency",
      fill = "MMSE Class"
    ) 
}

all_cont_plots <- map(continous, ~ cont_plot(df, .x))
all_cont_plots



#Null Value Handling:
df <- df %>% select(-ID)
colnames(df)
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
  Hyperlipidemia=factor(Hyperlipidemia,
                         levels = c(0,1),
                         labels = c('No','Yes')
  ),
  MMSE_class=factor(MMSE_class,
                    levels = c(0,1),
                    labels = c('Not at risk','At risk')
  )
)
str(df)

 

corr = df %>% select_if(is.numeric) %>% 
  cor(., method = "spearman") %>% round(3) 

corr %>% write.csv("corr.csv") 

corr
library(corrplot)
corrplot(corr, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

df_scale <- df %>% mutate(across(all_of(continous), ~ as.numeric(scale(.x)))) 
df_scale
df_clean <- df

library(caret)

set.seed(123)
train_index <- sample(nrow(df_scale),0.7 * nrow(df_scale))

train_clean <- df_clean[train_index, ]
test_clean  <- df_clean[-train_index, ]

train_scaled <- df_scale[train_index, ]
test_scaled  <- df_scale[-train_index, ]


#model building

library(randomForest)
library(e1071)
library(pROC)
library(class)




#random forest
set.seed(123)
rf <- randomForest(MMSE_class ~ ., data = train_clean)
rf_pred <- predict(rf, test_clean)
rf_cm <- confusionMatrix(rf_pred, test_clean$MMSE_class)
rf_cm

#svm
set.seed(123)
svm <- svm(MMSE_class ~ ., data = train_scaled,
                kernel = "radial", probability = TRUE)
svm_pred <- predict(svm, test_scaled, probability = TRUE)
svm_cm <- confusionMatrix(svm_pred, test_scaled$MMSE_class)
svm_cm


# RF baseline
rf_acc <- as.numeric(rf_cm$overall["Accuracy"])
rf_prec <- as.numeric(rf_cm$byClass["Precision"])
rf_rec <- as.numeric(rf_cm$byClass["Recall"])
rf_f1 <- as.numeric(rf_cm$byClass["F1"])

# SVM baseline
svm_acc <- as.numeric(svm_cm$overall["Accuracy"])
svm_prec <- as.numeric(svm_cm$byClass["Precision"])
svm_rec <- as.numeric(svm_cm$byClass["Recall"])
svm_f1 <- as.numeric(svm_cm$byClass["F1"])


# RF probabilities
rf_prob <- predict(rf, test_clean, type = "prob")[, "At risk"]

# SVM probabilities
svm_prob <- attr(svm_pred, "probabilities")[, "At risk"]

# ROC
rf_roc <- roc(test_clean$MMSE_class, rf_prob)
svm_roc <- roc(test_scaled$MMSE_class, svm_prob)

# AUC
rf_auc <- as.numeric(auc(rf_roc))
svm_auc <- as.numeric(auc(svm_roc))


# Create table
results <- data.frame(
  Model = c("Random Forest (Baseline)", "SVM (Baseline)"),
  Accuracy = round(c(rf_acc, svm_acc), 3),
  Precision = round(c(rf_prec, svm_prec), 3),
  Recall = round(c(rf_rec, svm_rec), 3),
  F1_Score = round(c(rf_f1, svm_f1), 3),
  AUC = round(c(rf_auc, svm_auc), 3)
)

results




# Baseline ROC Curve
plot(rf_roc,
     col = "blue",
     lwd = 3,
     main = "ROC Curve - SVM and Random Forest Model",
     legacy.axes = TRUE)

plot(svm_roc,
     col = "red",
     lwd = 3,
     add = TRUE)

abline(a = 0, b = 1, lty = 2, col = "gray")

legend("bottomright",
       legend = c(
         paste("Random Forest (AUC =", round(rf_auc, 3), ")"),
         paste("SVM (AUC =", round(svm_auc, 3), ")")
       ),
       col = c("blue", "red"),
       lwd = 3,
       bty = "n")

