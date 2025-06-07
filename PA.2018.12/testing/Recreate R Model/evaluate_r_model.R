# Script to evaluate the R model on our test data
library(reticulate)

# Log-likelihood function for Poisson model
LLfunction <- function(targets, predicted_values){
  p_v_zero <- ifelse(predicted_values <= 0, 0, predicted_values)
  p_v_pos <- ifelse(predicted_values <= 0, 0.000001, predicted_values)
  return(sum(targets*log(p_v_pos)) - sum(p_v_zero))
}

# Load the trained GLM model
load_model <- function() {
  # Read in data files for training
  data.all <- read.csv("../MSHA_Mine_Data_2013-2016.csv")
  
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
  
  return(list(model = GLM_Final, train_data = data.reduced4))
}

# Load Python test data using reticulate
load_test_data <- function() {
  # Use reticulate to import Python modules
  cat("Loading Python modules...\n")
  sys <- import("sys")
  os <- import("os")
  
  # Add the project directory to Python's path
  script_dir <- getwd()
  project_dir <- dirname(script_dir)
  sys$path$append(project_dir)
  sys$path$append(file.path(project_dir, "advanced_implementation", "scripts"))
  
  # Import the enhanced_feature_engineering module
  cat("Importing enhanced_feature_engineering...\n")
  tryCatch({
    enhanced_feature_engineering <- import("enhanced_feature_engineering")
    cat("Successfully imported enhanced_feature_engineering\n")
  }, error = function(e) {
    cat("Error importing enhanced_feature_engineering:", e$message, "\n")
    stop("Failed to import enhanced_feature_engineering")
  })
  
  # Get the test data
  cat("Loading test data...\n")
  result <- enhanced_feature_engineering$engineer_enhanced_features()
  train_data <- result[[1]]
  test_data <- result[[2]]
  
  # Extract the raw features we need for the R model
  raw_features <- test_data$raw_features
  
  # Convert to R dataframe
  test_df <- as.data.frame(raw_features)
  
  # Add target variable
  test_df$NUM_INJURIES <- test_data$y
  
  # Create INJ_RATE_PER2K
  test_df$INJ_RATE_PER2K <- test_df$NUM_INJURIES/(test_df$EMP_HRS_TOTAL/2000)
  
  # Create multi-class target (0: no injury, 1: one injury, 2: two injuries, 3+: three or more injuries)
  test_df$injury_class <- ifelse(test_df$NUM_INJURIES >= 3, 3, test_df$NUM_INJURIES)
  
  return(list(test_df = test_df, y_multi = test_data$y_multi))
}

# Prepare test data for the R model
prepare_test_data <- function(test_df, model_data) {
  # Apply the same transformations as in the training data
  
  # Create MINE_CHAR
  test_df$MINE_CHAR <- paste(test_df$TYPE_OF_MINE, test_df$COMMODITY)
  test_df$MINE_CHAR <- factor(test_df$MINE_CHAR, levels = levels(model_data$MINE_CHAR))
  
  # Take log of AVG_EMP_TOTAL
  test_df$LOG_AVG_EMP_TOTAL <- log(test_df$AVG_EMP_TOTAL)
  test_df$AVG_EMP_TOTAL <- NULL
  
  # Create ADJ_STATUS if it exists in the model
  if ("ADJ_STATUS" %in% names(model_data)) {
    test_df$ADJ_STATUS <- "Open"  # Default value
    test_df$ADJ_STATUS <- factor(test_df$ADJ_STATUS, levels = levels(model_data$ADJ_STATUS))
  }
  
  return(test_df)
}

# Evaluate the model on test data
evaluate_model <- function() {
  cat("Loading R model...\n")
  model_result <- load_model()
  GLM_Final <- model_result$model
  train_data <- model_result$train_data
  
  cat("Loading test data...\n")
  test_result <- load_test_data()
  test_df <- test_result$test_df
  
  cat("Preparing test data for R model...\n")
  test_df_prepared <- prepare_test_data(test_df, train_data)
  
  # Generate predictions
  cat("Generating predictions...\n")
  glm.predict <- tryCatch({
    predict(GLM_Final, newdata = test_df_prepared, type = "response")
  }, error = function(e) {
    cat("Error in prediction:", e$message, "\n")
    cat("Missing variables in test data:", setdiff(names(train_data), names(test_df_prepared)), "\n")
    cat("Extra variables in test data:", setdiff(names(test_df_prepared), names(train_data)), "\n")
    stop("Failed to generate predictions")
  })
  
  # Calculate log-likelihood
  ll_final <- LLfunction(test_df_prepared$NUM_INJURIES, glm.predict)
  cat("Log-likelihood:", round(ll_final, 3), "\n")
  
  # Show distribution of actual injury counts
  injury_table <- table(test_df_prepared$NUM_INJURIES)
  cat("\nDistribution of injury counts:\n")
  print(injury_table)
  
  # Show how many mines have 0, 1, 2, 3+ injuries
  cat("\nMines with 0 injuries:", sum(test_df_prepared$NUM_INJURIES == 0), "\n")
  cat("Mines with 1 injury:", sum(test_df_prepared$NUM_INJURIES == 1), "\n")
  cat("Mines with 2 injuries:", sum(test_df_prepared$NUM_INJURIES == 2), "\n")
  cat("Mines with 3+ injuries:", sum(test_df_prepared$NUM_INJURIES >= 3), "\n")
  
  # Calculate prediction accuracy by injury count class
  # Create injury count classes: 0, 1, 2, 3+
  actual_class <- test_df_prepared$injury_class
  pred_rounded <- round(glm.predict)
  pred_class <- ifelse(pred_rounded >= 3, 3, pred_rounded)
  
  # Create confusion matrix
  conf_matrix <- table(Actual = actual_class, Predicted = pred_class)
  cat("\nConfusion Matrix:\n")
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
}

# Run the evaluation
tryCatch({
  cat("Starting evaluation...\n")
  evaluate_model()
  cat("Evaluation completed successfully\n")
}, error = function(e) {
  cat("Error during evaluation:", e$message, "\n")
  print(e)
  traceback()
})
