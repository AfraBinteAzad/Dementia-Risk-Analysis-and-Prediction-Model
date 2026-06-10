install.packages('tidyverse')
install.packages('Amelia')
install.packages('readxl')
install.packages('caret')
install.packages('pROC')
install.packages('corrplot')
library(tidyverse)
library(Amelia)
library(readxl)
library(caret)
library(pROC)
library(corrplot)
library(randomForest)
df <- read_excel("C:/Users/User/Desktop/Dementia Risk Prediction Model/dataset.xlsx")

head(df)
colnames(df)

categorical <- c("Education_ID", "Mobility", "MNAa_q3", "Hyperlipidemia", "MMSE_class")
continous   <- c("Age", "body_height", "body_weight", "MNAb_tot", "waist", "MNAa_tot")

# Frequency tables for categories
df %>% select(all_of(categorical)) %>% map(table)

# Continuous Summary Statistics
df %>% select(all_of(continous)) %>%
  map(., ~{
    c(
      Mean = mean(.x, na.rm = TRUE) %>% round(2),
      SD   = sd(.x, na.rm = TRUE) %>% round(2)
    )
  })

# Categorical value distribution graph function
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
all_cat_plots # Display plots

# Continuous value distribution graph function
cont_plot <- function(data, conti) {
  data %>%
    ggplot(aes(
      x = factor(MMSE_class),
      y = .data[[conti]],
      fill = factor(MMSE_class)
    )) +
    geom_boxplot() +
    labs(
      title = paste("Boxplot of", conti),
      x = "MMSE Class",
      y = conti,
      fill = "MMSE Class"
    )
}

all_cont_plots <- map(continous, ~ cont_plot(df, .x))
all_cont_plots 


# 4. Null Value Handling & Imputation

df <- df %>% select(-ID)

#missing data analysis
colSums(is.na(df))
round(colMeans(is.na(df)) * 100, 2)
total_missingness <- sum(is.na(df)) / (nrow(df) * ncol(df)) * 100
total_missingness

#overall missingness 
missing_df <- data.frame(
  Variable = names(df),
  MissingPercent = round(colMeans(is.na(df)) * 100, 2)
)

ggplot(missing_df, aes(x = reorder(Variable, -MissingPercent), y = MissingPercent)) +
  geom_col(fill = "steelblue") +
  labs(title = "Percentage of Missing Values per Variable", x = "Variable", y = "Missing (%)")

missmap(df, main = "Missing Map", col = c('yellow', 'black'), Legend = FALSE)

# Clean missing target values
df <- df %>% filter(!is.na(MMSE_class))
colSums(is.na(df))

# median imputation for continuous values
for(i in continous){
  df[[i]][is.na(df[[i]])] <- median(df[[i]], na.rm = TRUE)
}

# mode function for categorical values
get_mode <- function(v){
  uniq <- na.omit(unique(v))
  uniq[which.max(tabulate(match(v, uniq)))]
}

# mode imputation for categorical values
for(i in categorical){
  df[[i]][is.na(df[[i]])] <- get_mode(df[[i]])
}

# confirm clean data
round(colMeans(is.na(df)) * 100, 2)
missmap(df, main = "Missing Map", col = c('yellow', 'black'), Legend = FALSE)

#data factorization & feature Scaling

df <- df %>% mutate(
  Education_ID = factor(Education_ID,
                        levels = c(1, 2, 3, 4),
                        labels = c('No formal education', 'Elementary school level', 'High school level', 'College level or above')),
  Mobility     = factor(Mobility,
                        levels = c(1, 2, 3, 4),
                        labels = c('Wheelchair', 'Walker', 'Cane', 'Self-mobility')),
  MNAa_q3      = factor(MNAa_q3,
                        levels = c(0, 1, 2),
                        labels = c('Bedridden', 'Cannot go out', 'Independent')),
  Hyperlipidemia = factor(Hyperlipidemia,
                          levels = c(0, 1),
                          labels = c('No', 'Yes')),
  MMSE_class   = factor(MMSE_class,
                        levels = c(1, 0),
                        labels = c('At_risk', 'Not_at_risk'))
)
str(df)

#spearman correlation matrix for continous variables
corr <- df %>% select_if(is.numeric) %>% 
  cor(., method = "spearman") %>% round(3) 

corr %>% write.csv("corr.csv") 
corr

corrplot(corr, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

#scale data
df_clean <- df
df_scale <- df %>% mutate(across(all_of(continous), ~ as.numeric(scale(.x)))) 
df_scale


#data splittinf
set.seed(123) 
train_index  <- sample(nrow(df_scale), 0.7 * nrow(df_scale)) 

train_clean  <- df_clean[train_index, ] 
test_clean   <- df_clean[-train_index, ] 

train_scaled <- df_scale[train_index, ] 
test_scaled  <- df_scale[-train_index, ]

#k fold validation
cv_control <- trainControl(
  method          = "repeatedcv",     
  number          = 5,                
  repeats         = 3,                
  summaryFunction = twoClassSummary,  
  classProbs      = TRUE,             
  sampling        = "down",           
  savePredictions = "final"           
)


#model building

#random forest train
set.seed(123)
rf_model <- train(
  MMSE_class ~ ., 
  data      = train_clean, 
  method    = "rf", 
  trControl = cv_control,
  metric    = "ROC"
)


#svm train
set.seed(123)
svm_model <- train(
  MMSE_class ~ ., 
  data      = train_scaled, 
  method    = "svmRadial", 
  trControl = cv_control,
  metric    = "ROC"
)

rf_model
svm_model


#model evaluation

# predict classes and probabilities on test set
rf_pred  <- predict(rf_model, test_clean)
rf_prob  <- predict(rf_model, test_clean, type = "prob")[, "At_risk"]
rf_cm    <- confusionMatrix(rf_pred, test_clean$MMSE_class, positive = "At_risk")

svm_pred <- predict(svm_model, test_scaled)
svm_prob <- predict(svm_model, test_scaled, type = "prob")[, "At_risk"]
svm_cm   <- confusionMatrix(svm_pred, test_scaled$MMSE_class, positive = "At_risk")


rf_metrics  <- c(rf_cm$overall["Accuracy"], rf_cm$byClass[c("Precision", "Recall", "F1")])
svm_metrics <- c(svm_cm$overall["Accuracy"], svm_cm$byClass[c("Precision", "Recall", "F1")])

#ROC + AUC
rf_roc  <- roc(test_clean$MMSE_class, rf_prob, levels = c("Not_at_risk", "At_risk"))
svm_roc <- roc(test_scaled$MMSE_class, svm_prob, levels = c("Not_at_risk", "At_risk"))

rf_auc  <- as.numeric(auc(rf_roc))
svm_auc <- as.numeric(auc(svm_roc))

#Summary Table
results <- data.frame(
  Model     = c("Random Forest", "SVM"),
  Accuracy  = round(c(rf_metrics["Accuracy"], svm_metrics["Accuracy"]), 3),
  Precision = round(c(rf_metrics["Precision"], svm_metrics["Precision"]), 3),
  Recall    = round(c(rf_metrics["Recall"], svm_metrics["Recall"]), 3),
  F1_Score  = round(c(rf_metrics["F1"], svm_metrics["F1"]), 3),
  AUC       = round(c(rf_auc, svm_auc), 3)
)

results


#ROC Curve Plotting

plot(rf_roc, col = "blue", lwd = 3, main = "ROC Curve - Cross-Validated Models", legacy.axes = TRUE)
plot(svm_roc, col = "red", lwd = 3, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")

legend("bottomright", 
       legend = c(
         paste("Random Forest (AUC =", round(rf_auc, 3), ")"), 
         paste("SVM (AUC =", round(svm_auc, 3), ")")
       ), 
       col = c("blue", "red"), lwd = 3, bty = "n")


