### Noam Kadosh
### Kaggle Competition
### Titanic - Machine Learning from Disaster
### https://github.com/noamkadosh

################################################
# Titanic Survival Prediction Project #
################################################

### Loading libraries ###
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(mice)) install.packages("mice", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(ada)) install.packages("ada", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")



# Loading the dataset
train <- read_csv("Data/train.csv")
test <- read.csv("Data/test.csv")

test$Survived <- NA
titanic <- rbind(train, test)

summary(titanic)

### Feature Engineering ###
# The title feature
titanic$Title <- gsub("(.*, )|(\\..*)", "", titanic$Name)

titanic %>%
  group_by(Title) %>%
  summarize(count = n())

titanic$Title[titanic$Title == "Ms"] <- "Miss"
titanic$Title[titanic$Title == "Mlle"] <- "Miss"
titanic$Title[titanic$Title == "Mme"] <- "Mrs"
other_title <- c("Capt", "Col", "Dr", "Major", "Don", "Dona", "Jonkheer", "Lady", "Rev", "Sir", "the Countess")
titanic$Title[titanic$Title %in% other_title] <- "Other_Title"

# Family name feature
titanic$Family <- gsub("(,.*)", "", titanic$Name)

titanic %>%
  group_by(Family) %>%
  summarize(count = n(), .groups = "drop") %>%
  slice_max(count, n = 10)

# Family size feature
titanic$Group <- titanic$Parch + titanic$SibSp + 1
titanic$FamilySize <- 0
titanic$FamilySize[which(titanic$Group < 2)] <- "single"
titanic$FamilySize[which(titanic$Group < 5 & titanic$Group > 1)] <- "small"
titanic$FamilySize[which(titanic$Group > 4)] <- "large"

# Adding people on ticket feature
passengers_on_ticket <- titanic %>%
  select(Ticket) %>%
  group_by(Ticket) %>%
  summarize(PassOnTckt = n(), .groups = "drop")

titanic <- left_join(titanic, passengers_on_ticket, by = "Ticket")

# Adding fare per person feature
titanic$FarePerPerosn <- titanic$Fare / titanic$PassOnTckt

### Filling Missing Values ###
# Deck
titanic$Deck <- substring(titanic$Cabin, 1, 1)
titanic$Deck[is.na(titanic$Cabin)] <- "U"
titanic$Deck[which(titanic$Deck == "")] <- "U"

# Look for the NAs in the Fare column
titanic %>%
  filter(is.na(Fare))

# Density plot to see what are the prices for those who embarked from S and are in third class.
third_class_s <- titanic %>%
  filter(Pclass == 3 & Embarked == "S")
third_class_s %>%
  ggplot(aes(Fare)) +
  geom_density(color = "black", fill = "#2e4057", alpha = 0.6, na.rm = TRUE) +
  scale_x_continuous(name = "Fare", label = scales::dollar_format(), breaks = c(0, median(third_class_s$Fare, na.rm = TRUE), 20, 30, 40, 50, 60)) +
  geom_vline(aes(xintercept = median(Fare, na.rm = TRUE)), colour='red', lwd = 0.25) +
  ggtitle("Fare Density")

# Assign the median to the NA in the fare column.
i <- which(is.na(titanic$Fare))
titanic$Fare[i] <- median(third_class_s$Fare, na.rm = TRUE)
titanic$FarePerPerosn[i] <- titanic$Fare[i] / titanic$PassOnTckt[i]

# Look for the NAs in the Embarked column
titanic %>%
  filter(is.na(Embarked))

first_class <- titanic %>%
  filter(Pclass == 1 & !is.na(Embarked))
first_class %>%
  ggplot(aes(x = Embarked, y = FarePerPerosn)) +
  geom_boxplot(na.rm = TRUE) +
  geom_hline(aes(yintercept = 40), colour='red', lwd = 0.25) +
  scale_y_continuous(name = "Fare", labels = scales::dollar_format(), breaks = c(0, 10, 20, 30, 40, 50, 75, 100, 125, 150)) +
  xlab("Embarked") +
  ggtitle("Median Fare per Embarkment Location")

# Looks like C
first_class %>%
  filter(FarePerPerosn < 41 & FarePerPerosn > 39) %>%
  ggplot(aes(Embarked)) +
  geom_bar(color = "black", fill = "#2e4057", alpha= 0.6) +
  scale_y_continuous(limits = c(0, 12), breaks = c(1, 3, 5, 7, 9, 11))

# Looking at the prices between 39 and 41 most of them are labeled C, so our NAs will be assigned C
titanic$Embarked[which(is.na(titanic$Embarked))] <- "C"

titanic <- titanic %>% mutate_if(is.character, as.factor)
titanic$Pclass <- as.factor(titanic$Pclass)

# Predicting Age with MICE
mice_data <- titanic %>%
  select(!c('PassengerId','Survived','Name','Ticket','Cabin','Family', 'Fare'))
mice_output <- mice(mice_data, method = "rf")
mice_output <- complete(mice_output)

titanic %>%
  ggplot(aes(Age)) +
  geom_histogram(binwidth = 2, color = "black", fill = "#2e4057", alpha= 0.6, na.rm = TRUE) +
  ggtitle("Original")

mice_output %>%
  ggplot(aes(Age)) +
  geom_histogram(binwidth = 2, color = "black", fill = "#2e4057", alpha= 0.6, na.rm = TRUE) +
  ggtitle("MICE")

titanic$Age <- mice_output$Age

# Child/Adult feature
titanic %>%
  summarize(Child = ifelse(Age < 18, "Child", "Adult")) %>%
  ggplot(aes(Child)) +
  geom_bar()

titanic <- titanic %>%
  mutate(Child = ifelse(Age < 18, "Child", "Adult"))

# The percentage of family survived feature
titanic_temp <- titanic %>%
  filter(Group > 2 & !is.na(Survived))
titanic_temp$percentFamilySurvived <- ave(titanic_temp$Survived, titanic_temp$Family)
titanic <- left_join(titanic, titanic_temp)

titanic_temp <- titanic %>% filter(Group > 2) %>%
  group_by(Family) %>%
  summarize(percentFam = mean(percentFamilySurvived, na.rm = TRUE))

titanic$percentFamilySurvived <- ifelse(titanic$Family %in% titanic_temp$Family & titanic$Group > 1, titanic_temp$percentFam, -1)
titanic$percentFamilySurvived[which(is.nan(titanic$percentFamilySurvived))] <- mean(titanic$percentFamilySurvived[which(titanic$percentFamilySurvived > -1)])

summary(titanic)

train <- titanic[which(!is.na(titanic$Survived)),]
train$Survived <- as.factor(train$Survived)
test <- titanic[which(is.na(titanic$Survived)),]

set.seed(9, sample.kind="Rounding")
# Logistic Regression - 0.76315
control <- trainControl(method = "cv", number = 10)
glm_fit <- train(Survived ~ Pclass + Sex + Age + Title + FamilySize + PassOnTckt + FarePerPerosn + Child + percentFamilySurvived,
                 method = "glm", data = train, trControl = control)

glm_predictions <- predict(glm_fit, test)
plot(glm_predictions)

# Random Forest - 0.76076
tune <- data.frame(mtry = seq(2, 6, 1))
rf_fit <- train(Survived ~ Pclass + Sex + Age + Title + FamilySize + PassOnTckt + FarePerPerosn + Child + percentFamilySurvived,
                 method = "rf", data = train, trControl = control, tuneGrid = tune)
ggplot(rf_fit) +
  geom_line()

rf_predictions <- predict(rf_fit, test)
plot(rf_predictions)

importance <- importance(rf_fit$finalModel)
importance

# Gradient Boost - 0.77033
gbm_fit <- train(Survived ~ Pclass + Sex + Age + Title + FamilySize + PassOnTckt + FarePerPerosn + Child + percentFamilySurvived,
                 method = "gbm", data = train, trControl = control)
ggplot(gbm_fit) +
  geom_line()

gbm_predictions <- predict(gbm_fit, test)
plot(gbm_predictions)

# XGBoost - 0.77751
xgboost_fit <- train(Survived ~ Pclass + Sex + Age + Title + Group + FarePerPerosn + Child + percentFamilySurvived,
                 method = "xgbTree", data = train, trControl = control)
ggplot(xgboost_fit) +
  geom_line()

xgboost_predictions <- predict(xgboost_fit, test)
plot(xgboost_predictions)

# ANN - 0.76076
tune <- expand.grid(size = seq(24, 28, len = 2), decay = c(0.1, 0.01, 0.001, 0.0001))
neuralnet_fit <- train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + PassOnTckt + FarePerPerosn + Child + percentFamilySurvived,
                     method = "nnet", data = train, trControl = control, tuneGrid = tune)
ggplot(neuralnet_fit) +
  geom_line()

neuralnet_predictions <- predict(neuralnet_fit, test)
plot(neuralnet_predictions)

# Majority Ensemble - 0.77033
ensemble_predictions <- tibble(as.numeric(levels(glm_predictions)[glm_predictions]), as.numeric(levels(rf_predictions)[rf_predictions]), 
                               as.numeric(levels(gbm_predictions)[gbm_predictions]), as.numeric(levels(xgboost_predictions)[xgboost_predictions]),
                               as.numeric(levels(neuralnet_predictions)[neuralnet_predictions]))
ensemble_predictions <- ensemble_predictions %>%
  mutate(predictions = round(rowMeans(across(where(is.numeric)))))

predictions <- data.frame(PassengerID = test$PassengerId, Survived = xgboost_predictions)
write.csv(predictions, file = 'titanic_predictions.csv', row.names = F)
# Add more models and examine them