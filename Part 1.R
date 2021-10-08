# Loan Prediction Based on Customer Behavior
# Contributors: Xiaohang Li - Data Undertstanding, Data Preparation; 
#               Michelle Fu - Data Preparation, Modelling, Evaluation; 
#               Jiamin Feng - Data Preparation

# Load the packages
source("DataAnalyticsFunctions.R")
install.packages('dplyr')
install.packages('plyr')
installpkg("partykit")
installpkg("libcoin")
installpkg("randomForest")
installpkg("tree")
install.packages("ROSE")
install.packages('rpart')
install.packages('rattle')
install.packages('fastDummies')
install.packages('DAAG')
install.packages('party')
install.packages('mlbench')
install.packages('caret')
install.packages('pROC')
install.packages('precrec')
installpkg("glmnet")
installpkg("cvAUC")
install.packages('rfUtilities')
install.packages("RColorBrewer")
install.packages("formattable")
library(formattable)
library(RColorBrewer)
library(glmnet)
library(plotROC)
library('plyr')
library('dplyr')
library('tidyr')
library('tree')
library(randomForest)
library(libcoin)
library(partykit)
library(ROSE)
library(rpart)
library(rattle)
library('fastDummies')
library(DAAG)
library(party)
library(mlbench)
library(caret)
library(pROC)
library(precrec)
library(cvAUC)
library(rfUtilities)
library(mclust)
library(corrplot)

# Load the dataset 
data <- read.csv("Data.csv")

###########################################################
### Data Understanding - Xiaohang Li

# Profession Count Bar plot
coul <- brewer.pal(3, "Dark2") 
count <- table(data$Profession)
barplot(count, las=2, cex.names=.7,xlab = "Profession",ylab = "Count",col=coul,ylim =c(0,6000))

# Box plot : Age 
# Observations : Mean age is around 50. Maximum age: 79, Minimum age: 21.
Age <- data$Age
summary(Age)
boxplot(Age, main ="Box Plot for Age", ylab = "Age")

# Pie Chart - House ownership
# Observations: Approximately 92% of whole population rent house. 5% own house. 3% neither own or rent
df <- as.data.frame(table(data$House_Ownership))
df <- as.data.frame(df %>% rename(House = Var1))
# Compute percentages
df$fraction = percent(df$Freq / sum(df$Freq))

# Hole size
hsize <- 3
df <- df %>% 
  mutate(x = hsize)

# Plot the pie chart
ggplot(df, aes(x = hsize, y = Freq, fill = House)) +
  geom_col(color = "black") +
  geom_text(aes(label = fraction),
            position = position_stack(vjust = 0.5)) +
  coord_polar(theta = "y") +
  scale_fill_brewer(palette = "GnBu") +
  xlim(c(0.2, hsize + 0.5)) +
  theme(panel.background = element_rect(fill = "white"),
        panel.grid = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank())

# Pie Chart - Car ownership
# Observations: Majority (70%) of the population does not own a car. 
df2 <- as.data.frame(table(data$Car_Ownership))
df2 <- as.data.frame(df2 %>% rename(Car = Var1))

# Compute percentages
df2$fraction = percent(df2$Freq / sum(df2$Freq))

# Hole size
hsize <- 3
df2 <- df2 %>% 
  mutate(x = hsize)

#Plot the pie chart
ggplot(df2, main='Car ownership',aes(x = hsize, y = Freq, fill = Car)) +
  geom_col(color = "black") +
  geom_text(aes(label = fraction),
            position = position_stack(vjust = 0.5)) +
  coord_polar(theta = "y") +
  scale_fill_brewer(palette = "GnBu") +
  xlim(c(0.2, hsize + 0.5)) +
  theme(panel.background = element_rect(fill = "white"),
        panel.grid = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank())

# Pie Chart - Marriage status
# Observations:
df3 <- as.data.frame(table(data$Married.Single))
df3 <- as.data.frame(df3 %>% rename(Marriage = Var1))

# Compute percentages
df3$fraction = percent(df3$Freq / sum(df3$Freq))

# Hole size
hsize <- 3
df3 <- df3 %>% 
  mutate(x = hsize)

#Plot the pie chart
ggplot(df3, aes(x = hsize, y = Freq, fill = Marriage)) +
  geom_col(color = "black") +
  geom_text(aes(label = fraction),
            position = position_stack(vjust = 0.5)) +
  coord_polar(theta = "y") +
  scale_fill_brewer(palette = "RdYlGn") +
  xlim(c(0.2, hsize + 0.5)) +
  theme(panel.background = element_rect(fill = "white"),
        panel.grid = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank())

## Risk count for each age group
# Store Age and risk_flag into a data frame
agerisk <- as.data.frame(table(data$Age, data$Risk_Flag))
agerisk <- as.data.frame(agerisk %>% rename(age = Var1, risk = Var2))
# Set criteria for age group based on the quantile of age column in data
# Min.Age = 21, 1st quantile.Age = 35, Median.Age = 50, 
# 3rd quantile.Age = 65, Max.Age = 79
# According to Medicare, 65 years old or older is considered as senior citizen
summary(data$Age)
# Divide age into three groups: youth (age < 36), medium (35<age<64), 
# senior (64<age)
# Set four age group criteria
youth <- c(1:35)
medium <- c(36:64)
senior <- c(65:79)
# Divide data set into each age group
agerisk <- agerisk %>% 
  mutate(
    Agegroup = case_when(
      age %in% youth ~ "youth",
      age %in% medium  ~ "medium",
      age %in% senior ~ "senior",
      TRUE ~ "other"
    )
  )
# Drop age column
grouprisk <- subset(agerisk,select=c("risk","Freq","Agegroup"))
# Add up total people based on risk type and age group
group_risk <- aggregate(cbind(Freq)~ Agegroup + risk, data=grouprisk,sum)

# Organize data
r0 <- subset(group_risk, risk == 0)
r1 <- subset(group_risk, risk == 1)
r <- merge(r0,r1, by = "Agegroup")

# Form new data frame for bar plot
RISK <- c("NO", "YES")
youth_age <- c(56793,8933)
medium_age <- c(107551,14531)
senior_age <- c(56660,7532)
gr <- data.frame(RISK, youth_age, medium_age,senior_age)

# Plot the bar chart
barplot(as.matrix(gr),main = "Risk Count of Each Agegroup",xlab = "Class",
        col=c("yellow","Red"))
legend("topleft",
       c("Risk","No Risk"),
       fill = c("red","yellow")
)

## Attempting PCA for numeric variables 
library(ggbiplot)
pca <- prcomp(data[,1:3,10:12],center=TRUE,scale. = TRUE)
ggbiplot(pca) # Plot PCA
# PCA does not work because the data is discrete and made up by human.
# PCA does not help when you have categorical correlated features 
# as you will be using 0 and 1 for values so it does not know how good they are.

#Attempting MCA with profession
require(FactoMineR)
require(ggplot2)
# select categorical columns
newdata <- data[,c("Married.Single","House_Ownership","Car_Ownership","Profession")]

# number of categories per variable
cats <- apply(newdata, 2, function(x) nlevels(as.factor(x)))
cats

# apply MCA
mca1 = MCA(newdata, graph = FALSE)

# list of results
mca1
# table of eigenvalues
mca1$eig
# data frame with variable coordinates
mca1_vars_df = data.frame(mca1$var$coord, Variable = rep(names(cats), cats))

# data frame with observation coordinates
mca1_obs_df = data.frame(mca1$ind$coord)

# plot of variable categories
ggplot(data=mca1_vars_df, 
       aes(x = Dim.1, y = Dim.2, label = rownames(mca1_vars_df))) +
  geom_hline(yintercept = 0, colour = "gray70") +
  geom_vline(xintercept = 0, colour = "gray70") +
  geom_text(aes(colour=Variable)) +
  ggtitle("MCA plot of variables using R package FactoMineR")
# MCA plot of observations and categories
ggplot(data = mca1_obs_df, aes(x = Dim.1, y = Dim.2)) +
  geom_hline(yintercept = 0, colour = "gray70") +
  geom_vline(xintercept = 0, colour = "gray70") +
  geom_point(colour = "gray50", alpha = 0.7) +
  geom_density2d(colour = "gray80") +
  geom_text(data = mca1_vars_df, 
            aes(x = Dim.1, y = Dim.2, 
                label = rownames(mca1_vars_df), colour = Variable)) +
  ggtitle("MCA plot of variables using R package FactoMineR") +
  scale_colour_discrete(name = "Variable")

#MCA without profession

# select categorical columns
newdata2 <- data[,c("Married.Single","House_Ownership","Car_Ownership")]

# number of categories per variable
cats2 <- apply(newdata2, 2, function(x) nlevels(as.factor(x)))
cats2

# apply MCA
mca2 = MCA(newdata2, graph = FALSE)

# list of results
mca2
# table of eigenvalues
mca2$eig
# data frame with variable coordinates
mca2_vars_df = data.frame(mca2$var$coord, Variable = rep(names(cats2), cats2))

# data frame with observation coordinates
mca2_obs_df = data.frame(mca2$ind$coord)

# plot of variable categories
ggplot(data=mca2_vars_df, 
       aes(x = Dim.1, y = Dim.2, label = rownames(mca2_vars_df))) +
  geom_hline(yintercept = 0, colour = "gray70") +
  geom_vline(xintercept = 0, colour = "gray70") +
  geom_text(aes(colour=Variable)) +
  ggtitle("MCA plot of variables using R package FactoMineR")

# MCA plot of observations and categories
ggplot(data = mca2_obs_df, aes(x = Dim.1, y = Dim.2)) +
  geom_hline(yintercept = 0, colour = "gray70") +
  geom_vline(xintercept = 0, colour = "gray70") +
  geom_point(colour = "gray50", alpha = 0.7) +
  geom_density2d(colour = "gray80") +
  geom_text(data = mca2_vars_df, 
            aes(x = Dim.1, y = Dim.2, 
                label = rownames(mca2_vars_df), colour = Variable)) +
  ggtitle("MCA plot of variables using R package FactoMineR") +
  scale_colour_discrete(name = "Variable")

## Try random forest and the importance graph to determine the importance  of each varibale


###########################################################
### Data Preparation - Michelle Fu, Xiaohang Li, Jiamin Feng

### I. Show summary of the dataset: 
### No missing values were discovered in any of the variables
summary(data) 

### II. Clean the dataset
### 1) Remove variables that do not provide values for prediction: Id, State
drops <- c("Id","STATE")
data <- data[,!(names(data) %in% drops)]
### 2) Replace the city variable by calculate income ratio
###    income ratio = personal income / average income in the city where the person locates
listOfMeans <- data %>%
  group_by(CITY) %>%
  summarise_at(vars(Income), list(name = "mean"))
DATA <- join(data, listOfMeans, by = "CITY")
DATA$IncomeRatio <- as.numeric(DATA$Income / DATA$name)
DATA <- DATA[,!(names(DATA) %in% c("name","CITY"))]
### 3) Convert columns into the correct data types
DATA$Income <- as.numeric(DATA$Income)
DATA$Age <- as.numeric(DATA$Age)
DATA$Experience <- as.numeric(DATA$Experience)
DATA$CURRENT_JOB_YRS <- as.numeric(DATA$CURRENT_JOB_YRS)
DATA$CURRENT_HOUSE_YRS <- as.numeric(DATA$CURRENT_HOUSE_YRS)

### III. Remove the duplicate records from the dataset
DATA <- DATA[!duplicated(DATA),]
summary(DATA)

### IV. Remove the unreasonable records 
### The legal age to work in India is 14, while in some of the records the difference
### between age and experience or current job years is smaller than 14
DATA <- DATA %>%
  filter((Age-Experience>=14) & (Age-CURRENT_JOB_YRS>=14))

### IV. under-sample the majority and over-sample the minority to solve the issue 
### of imbalanced classification
### Check target variables distribution - positives = 12%, negatives = 88%
### The data is very imbalanced and needs to be handled before machine learning 
### algorithm can be applied 
table(DATA$Risk_Flag)
prop.table(table(DATA$Risk_Flag))
### under-sampling & over-sample
DATA <- ovun.sample(Risk_Flag ~ ., data = DATA, method = "both", N = 50000, seed = 1)$data
table(DATA$Risk_Flag)
prop.table(table(DATA$Risk_Flag))

### DATA is now ready for use in further analysis
set.seed(42)
rows <- sample(nrow(DATA))
DATA <- DATA[rows, ]
write.csv(DATA, "PythonData.csv")

## Further analysis for modeling
### Construct dummy variables for non-numeric variables
Dum_DATA <- dummy_cols(DATA, select_columns = c('Married.Single', 'House_Ownership', 'Car_Ownership', "Profession"),
                       remove_selected_columns = TRUE)
Dum_DATA <- Dum_DATA[!names(Dum_DATA) 
                     %in% c("Profession_Web_designer",
                            "Married.Single_single",
                            "Car_Ownership_yes",
                            "House_Ownership_rented"),]

plot(factor(Risk_Flag) ~ ., data=Dum_DATA, col=c(8,2), ylab="Risk_Flag") 

## Correlation Matrix 
## There are no two linearly related variables in the dataset.
## We need to use a classification model instead of a regression model in the future modeling process. 

# Keep Dum_DATA with only numeric value
drops <- c("Married.Single","House_Ownership","Car_Ownership","Profession","CITY")
corrd <- DATA[,!(names(DATA) %in% drops)]
# calculate the correlation between each variable
corr <- cor(corrd)
# Plot the Correlation Matrix
corrplot(corr, method="number",col=brewer.pal(n=8, name="PuOr"))

## Model Based Clustering 1 - Experience, Income Ratio
# Prepare data
mydata <- subset(Dum_DATA,select=c(Experience, IncomeRatio))
s <- scale(mydata)
#fit data with best model and plot clustering
fit <- Mclust(s)
plot(fit) # plot results (classification model)
summary(fit) # display the best model

## Model Based Clustering 2 - Age, Income 
# Prepare data
mydata3 <- subset(Dum_DATA,select=c(Age, Income))
s3 <- scale(mydata3)
#fit data with best model and plot clustering
fit3 <- Mclust(s3)
plot(fit3) # plot results (density model)
summary(fit3) # display the best model



###########################################################
### Modelling - Michelle Fu

### 1) Logistic regression
### Constructed a simple logistic regression model with Risk_Flag as the dependent 
### variable and all the other variables as the independent variables. CITY was removed
### from model since it is a proxy for IncomeRatio. 
log_reg <- glm(Risk_Flag~., data=DATA, family="binomial")
summary(log_reg)
log_reg.pred <- predict(log_reg,DATA[,!names(DATA) %in% c("Risk_Flag")],type = "response")

# R2:0.003456889
R2(DATA$Risk_Flag, log_reg.pred)

# Area under the curve: 0.5335429
AUC(log_reg.pred, DATA$Risk_Flag)
rocplot <- ggplot(DATA,aes(m = log_reg.pred, d = Risk_Flag))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 

### 2）Logistic regression with interaction
### Constructed a logistic regression model with Risk_Flag as the dependent 
### variable and all the other variables as the independent variables. CITY was removed
### from model since it is a proxy for IncomeRatio. 
### Interaction terms were created by combining each pair of variables.
log_interact_reg <- glm(Risk_Flag~.^2, data=DATA, family="binomial")
summary(log_interact_reg)
log_interact_reg.pred <- predict(log_interact_reg,DATA[,!names(DATA) %in% c("Risk_Flag")],type = "response")

# R2:0.02929597
R2(DATA$Risk_Flag, log_interact_reg.pred)

# Area under the curve: 0.5942335
AUC(log_interact_reg.pred, DATA$Risk_Flag)
rocplot <- ggplot(DATA,aes(m = log_interact_reg.pred, d = Risk_Flag))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 


### 3) Logistic regression with Lasso
### Set up the data for Lasso 
Mx<- model.matrix(Risk_Flag ~ ., data=DATA)[,-1]
My<- DATA$Risk_Flag == 1

#### Calculate penalty parameter lambda
cv_model <- cv.glmnet(Mx, My, alpha = 1)
#plot(cv_model) 
best_lambda <- cv_model$lambda.min
best_lambda

### Constructed a  logistic regression Lasso model with Risk_Flag as the dependent 
### variable and all the other variables as the independent variables.
log_reg_lasso <- glmnet(Mx,My, family="binomial",lambda = best_lambda)
summary(log_reg_lasso)
coef(log_reg_lasso)
log_reg_lasso.pred <- predict(log_reg_lasso,Mx,type = "response")

# R2:0.003195569
R2(DATA$Risk_Flag, log_reg_lasso.pred)

# Area under the curve: 0.5326089
AUC(as.numeric(log_reg_lasso.pred), My)
rocplot <- ggplot(DATA,aes(m = log_interact_reg.pred, d = Risk_Flag))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 


### 4) Logistic regression with interaction with Lasso
### Set up the data for Lasso 
Mx_interact<- model.matrix(Risk_Flag ~ .^2, data=DATA)[,-1]
My_interact<- DATA$Risk_Flag == 1

#### Calculate penalty parameter lambda
cv_model_interact <- cv.glmnet(Mx_interact, My_interact, alpha = 1)
#plot(cv_model_interact) 
best_lambda_interact <- cv_model_interact$lambda.min
best_lambda_interact

### Constructed a  logistic regression Lasso model with Risk_Flag as the dependent 
### variable and all the other variables as the independent variables.
log_reg_interact_lasso <- glmnet(Mx_interact,My_interact, family="binomial",lambda = best_lambda_interact)
summary(log_reg_interact_lasso)
coef(log_reg_interact_lasso)
log_reg_interact_lasso.pred <- predict(log_reg_interact_lasso,Mx_interact,type = "response")

# R2:0.02678784
R2(DATA$Risk_Flag, log_reg_interact_lasso.pred)

# Area under the curve: 0.5909099
AUC(as.numeric(log_reg_interact_lasso.pred),My_interact)
rocplot <- ggplot(DATA,aes(m = log_reg_interact_lasso, d = Risk_Flag))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 

###########################################################
### Evaluation - Michelle Fu

### K Fold Cross Validation
### Create a vector of fold memberships (random order)
n<- nrow(DATA)
nfold <- 5
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

### Create an empty dataframe of results
OOS <- data.frame( logistic=rep(NA,nfold),logistic.interaction=rep(NA,nfold),
                   logistic.lasso=rep(NA,nfold),logistic.interaction.lasso=rep(NA,nfold)) 

### Use a for loop to run through the nfold trails
for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ## fit the four regressions and null model
  logistic <-glm(Risk_Flag==1~., data=DATA, subset=train, family="binomial")
  logistic.interaction <-glm(Risk_Flag==1~.^2, data=DATA, subset=train, family="binomial")
  logistic.lasso <- glmnet(Mx[train,],My[train],lambda = best_lambda,family="binomial")
  logistic.interaction.lasso <-glmnet(Mx_interact[train,],My_interact[train],lambda = best_lambda_interact,family="binomial")
  null <-glm(Risk_Flag==1~., data=DATA, subset=train,family="binomial")
  ## get predictions: type=response
  pred.logistic <- predict(logistic, newdata=DATA[-train,], type="response")
  pred.logistic.interaction <- predict(logistic.interaction, newdata=DATA[-train,], type="response")
  pred.logistic.lasso <- predict(logistic.lasso, newx=Mx[-train,],type="response")
  pred.logistic.interaction.lasso <- predict(logistic.interaction.lasso, newx=Mx_interact[-train,],type="response")
  pred.null <- predict(null, newdata=DATA[-train,], type="response")
  
  ## calculate and log AUC
  # Logistic
  OOS$logistic[k] <- AUC(pred.logistic,DATA$Risk_Flag[-train]==1)
  OOS$logistic[k]
  # Logistic Interaction
  OOS$logistic.interaction[k] <- AUC(pred.logistic.interaction,DATA$Risk_Flag[-train]==1)
  OOS$logistic.interaction[k]
  # Logistic Lasso
  OOS$logistic.lasso[k] <- AUC(pred.logistic.lasso, as.numeric(My[-train]==1))
  OOS$logistic.lasso[k]
  # Logistic Interaction Lasso
  OOS$logistic.interaction.lasso[k] <- AUC(pred.logistic.interaction.lasso, as.numeric(My_interact[-train]==1))
  OOS$logistic.interaction.lasso[k]
  #Null
  OOS$null[k] <- AUC(pred.null, DATA$Risk_Flag[-train]==1)
  OOS$null[k]
  
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}

### List the mean of the results stored in the dataframe OOS
colMeans(OOS)
m.OOS <- as.matrix(OOS)
rownames(m.OOS) <- c(1:nfold)
barplot(t(as.matrix(OOS)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=0.5),
        ylab= bquote( "Out of Sample " ~ AUC), xlab="Fold", names.arg = c(1:10))

### Plot a box blot to see how OOS AUC fluctuates across fold
if (nfold >= 10){
  names(OOS)[1] <-"logistic"
  boxplot(OOS, col="plum", las = 2, ylab=expression(paste("OOS ",AUC)), xlab="", main="10-fold Cross Validation")
  names(OOS)[1] <-"logistic"
}

################# Part 1 End ###############################
