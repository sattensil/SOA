# Log-likelihood function for Poisson model
LLfunction <- function(targets, predicted_values){
  p_v_zero <- ifelse(predicted_values <= 0, 0, predicted_values)
  p_v_pos <- ifelse(predicted_values <= 0, 0.000001 ,predicted_values)
  return(sum(targets*log(p_v_pos)) - sum(p_v_zero))
}

# Read in data files
data.all <- read.csv("../../MSHA_Mine_Data_2013-2016.csv")

# Data cleaning
data.nomissing <- data.all[!is.na(data.all$MINE_STATUS),]
data.nomissing <- data.nomissing[!is.na(data.nomissing$US_STATE),]
data.nomissing <- data.nomissing[!is.na(data.nomissing$PRIMARY),]

# Reduce variables
data.reduced <- data.nomissing
data.reduced$PRIMARY <- NULL
data.reduced$US_STATE <- NULL

# Create target variable
data.reduced$INJ_RATE_PER2K <- data.reduced$NUM_INJURIES/(data.reduced$EMP_HRS_TOTAL/2000)

# Remove closed or similar to closed mine status
no_good <- c("Closed by MSHA","Non-producing","Permanently abandoned","Temporarily closed")
data.reduced2 <- data.reduced
for (n in 1:length(no_good)){
  data.reduced2 <- data.reduced2[data.reduced2$MINE_STATUS != no_good[[n]], ]
}

# Remove low employee hours
data.reduced3 <- data.reduced2[data.reduced2$EMP_HRS_TOTAL >= 2000,]

# Resolve coal mining issue by combining categories
data.reduced3$ADJ_STATUS <- NULL
data.reduced3$ADJ_STATUS[data.reduced3$MINE_STATUS == "Active"] <- "Open"
data.reduced3$ADJ_STATUS[data.reduced3$MINE_STATUS == "Full-time permanent"] <- "Open"
data.reduced3$ADJ_STATUS[data.reduced3$MINE_STATUS == "Intermittent"] <- "Intermittent"
data.reduced3$ADJ_STATUS <- as.factor(data.reduced3$ADJ_STATUS)
data.reduced3$MINE_STATUS <- NULL

# Fix GLM data issues
data.reduced4 <- data.reduced3
data.reduced4$MINE_CHAR <- paste(data.reduced4$TYPE_OF_MINE, data.reduced4$COMMODITY)
data.reduced4$MINE_CHAR <- as.factor(data.reduced4$MINE_CHAR)
data.reduced4$MINE_CHAR <- relevel(data.reduced4$MINE_CHAR, ref="Sand & gravel Sand & gravel")

# Take log of AVG_EMP_TOTAL
data.reduced4$LOG_AVG_EMP_TOTAL <- log(data.reduced4$AVG_EMP_TOTAL)
data.reduced4$AVG_EMP_TOTAL <- NULL

# Final GLM model - exactly as in the solution
GLM_Final <- glm(NUM_INJURIES ~ SEAM_HEIGHT + PCT_HRS_UNDERGROUND + 
    PCT_HRS_MILL_PREP + PCT_HRS_OFFICE + MINE_CHAR + LOG_AVG_EMP_TOTAL + 
    LOG_AVG_EMP_TOTAL:PCT_HRS_UNDERGROUND + LOG_AVG_EMP_TOTAL:PCT_HRS_STRIP,
    family = poisson(),
    offset = log(EMP_HRS_TOTAL/2000),
    data = data.reduced4)

# Print model summary
print(summary(GLM_Final))

# Generate predictions
glm.predict <- predict(GLM_Final, newdata = data.reduced4, type = "response")

# Calculate log-likelihood
ll_final <- LLfunction(data.reduced4$NUM_INJURIES, glm.predict)
cat("Log-likelihood:", round(ll_final, 3), "\n")

# Show distribution of actual injury counts
injury_table <- table(data.reduced4$NUM_INJURIES)
print("Distribution of injury counts:")
print(injury_table)

# Show how many mines have 0, 1, 2, 3+ injuries
cat("\nMines with 0 injuries:", sum(data.reduced4$NUM_INJURIES == 0), "\n")
cat("Mines with 1 injury:", sum(data.reduced4$NUM_INJURIES == 1), "\n")
cat("Mines with 2 injuries:", sum(data.reduced4$NUM_INJURIES == 2), "\n")
cat("Mines with 3+ injuries:", sum(data.reduced4$NUM_INJURIES >= 3), "\n")

# Calculate prediction accuracy by injury count class
# Create injury count classes: 0, 1, 2, 3+
actual_class <- ifelse(data.reduced4$NUM_INJURIES >= 3, 3, data.reduced4$NUM_INJURIES)
pred_rounded <- round(glm.predict)
pred_class <- ifelse(pred_rounded >= 3, 3, pred_rounded)

# Create confusion matrix
conf_matrix <- table(Actual = actual_class, Predicted = pred_class)
print("\nConfusion Matrix:")
print(conf_matrix)

# Calculate class-specific accuracies
class_accuracies <- numeric(4)
for (i in 0:3) {
  class_accuracies[i+1] <- sum(actual_class == i & pred_class == i) / sum(actual_class == i)
}

cat("\nClass-specific accuracies:\n")
cat("Class 0 (No injuries):", round(class_accuracies[1], 3), "\n")
cat("Class 1 (1 injury):", round(class_accuracies[2], 3), "\n")
cat("Class 2 (2 injuries):", round(class_accuracies[3], 3), "\n")
cat("Class 3+ (3+ injuries):", round(class_accuracies[4], 3), "\n")

# Calculate overall accuracy
overall_accuracy <- sum(actual_class == pred_class) / length(actual_class)
cat("\nOverall accuracy:", round(overall_accuracy, 3), "\n")

# Effect of coefficients as in the solution
cat("\nEffect of a unit change in a predictor, in percent\n")
print(100*(exp(GLM_Final$coefficients) - 1))
