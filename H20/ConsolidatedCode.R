#Install PackagesPackages Needed
install.packages("randomForest")
install.packages("data.table")
install.packages("dplyr")
install.packages("h2o")
install.packages("dummies")

#Loading libraries

library(h2o)
library(data.table)
library(dplyr)
library(dummies)

h2o.shutdown()
#Initialize h20 instance
localH2O = h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
                    max_mem_size = "4G")  #max mem size is the maximum memory to allocate to H2O

#Setting file path
train_path<-"C:\\Users\\Deepak Kumar Sisodia\\Desktop\\R Project 2\\train.csv"
test_path<-"C:\\Users\\Deepak Kumar Sisodia\\Desktop\\R Project 2\\test.csv"

#Reading data into h2o 
train_data<-h2o.importFile( path = train_path
                            ,col.types = c("Factor","Factor","Factor","Factor","Factor","Factor","Factor"
                                          ,"Factor","Factor","Factor","Factor","Int")
                            )
test_data<-h2o.importFile( path = test_path
                            ,col.types = c("Factor","Factor","Factor","Factor","Factor","Factor","Factor"
                                          ,"Factor","Factor","Factor","Factor")
  
                          )

rm(train_path)
rm(test_path)
#________________________________________________________________________________

#Understanding the data
                          
#Viewing contents of h2o frame
str(train_data) #Get structure
colnames(train_data) #Get column names
head(train_data) #View first 6 rows - Also works with just entering the h20 frame name
summary(train_data,exact_quantiles=TRUE) #Summary statistics of variables
#To check if the data contains any factor columns
h2o.anyFactor(train_data)

#Plotting purchase value
h2o.hist(train_data$Purchase)

#Getting quantile values:
quantile(train_data$Purchase)

#Getting minimum and maximum values:
max(train_data$Purchase)
min(train_data$Purchase)

#To sort the data in ascending order of purchase
h2o.arrange(train_data,Purchase)                            
#________________________________________________________________________________
#Numeric conversion of age variable:
train_data$Age_numeric<-ifelse(train_data$Age=="0-17",0,ifelse(train_data$Age=="18-25",1,ifelse(train_data$Age=="26-35",2,ifelse(train_data$Age=="36-45",3,ifelse(train_data$Age=="46-50",4,ifelse(train_data$Age=="51-55",5,6))))))
#Numeric conversion of Gender variable:
train_data$Gender_numeric<-ifelse(train_data$Gender=="F",0,1)
#Numeric conversion of Stay in city years variable:
train_data$Stay_numeric<-ifelse(train_data$Stay_In_Current_City_Years=="4+",4,train_data$Stay_In_Current_City_Years)


#Convert age, gender, stay in current city years into numerical variables for test data
test_data$Age_numeric<-ifelse(test_data$Age=="0-17",0,ifelse(test_data$Age=="18-25",1,ifelse(test_data$Age=="26-35",2,ifelse(test_data$Age=="36-45",3,ifelse(test_data$Age=="46-50",4,ifelse(test_data$Age=="51-55",5,6))))))
test_data$Gender_numeric<-ifelse(test_data$Gender=="F",0,1)
test_data$Stay_numeric<-ifelse(test_data$Stay_In_Current_City_Years=="4+",4,test_data$Stay_In_Current_City_Years)

#________________________________________________________________________________

#Derived variables

#Get value fields for each categorical variable - mean purchase value and frequency
#Getting the values according to Marital_Status
ms<-h2o.group_by(train_data,by="Marital_Status",mean("Purchase"),nrow("Purchase"))
#Assigning column names
colnames(ms)<-c("Marital_Status","Mean_Purchase_MS","Count_MS")

#Getting values according to gender
gen<-h2o.group_by(train_data,by="Gender",mean("Purchase"),nrow("Purchase"))
#Assigning column names
colnames(gen)<-c("Gender","Mean_Purchase_Gen","Count_Gen")

#Getting values according to occupation
occ<-h2o.group_by(train_data,by="Occupation",mean("Purchase"),nrow("Purchase"))
#Assigning column names
colnames(occ)<-c("Occupation","Mean_Purchase_Occ","Count_Occ")

#Getting values according to city category
city<-h2o.group_by(train_data,by="City_Category",mean("Purchase"),nrow("Purchase"))
#Assigning column names
colnames(city)<-c("City_Category","Mean_Purchase_City","Count_City")

#Getting values according to Age
age<-h2o.group_by(train_data,by="Age",mean("Purchase"),nrow("Purchase"))
#Assiging column names
colnames(age)<-c("Age","Mean_Purchase_Age","Count_Age")

#Getting values according to Stay_In_Current_City_Years
stay<-h2o.group_by(train_data,by="Stay_In_Current_City_Years",mean("Purchase"),nrow("Purchase"))
#Assiging column names
colnames(stay)<-c("Stay_In_Current_City_Years","Mean_Purchase_Stay","Count_Stay")

#Getting values according to User_ID
user<-h2o.group_by(train_data,by="User_ID",mean("Purchase"),nrow("Purchase"))
#Assiging column names
colnames(user)<-c("User_ID","Mean_Purchase_User","Count_User")

#Getting values according to "Product_ID"
product<-h2o.group_by(train_data,by="Product_ID",mean("Purchase"),nrow("Purchase"))
#Assiging column names
colnames(product)<-c("Product_ID","Mean_Purchase_Product","Count_Product")

#Merging these fields with the train and test data using left join with the train and test data as the primary source.
#If it is a new factor, then the values that would be assigined to it is Na which can be converted into 0's to indicate no historical data
#In our data set the test data did not contain any new factor value
#Combine with train data

train_data<-h2o.merge(train_data,ms,by = "Marital_Status",all.x = T)
train_data<-h2o.merge(train_data,gen,by = "Gender",all.x = T)
train_data<-h2o.merge(train_data,occ,by = "Occupation",all.x = T)
train_data<-h2o.merge(train_data,city,by = "City_Category",all.x = T)
train_data<-h2o.merge(train_data,age,by = "Age",all.x = T)
train_data<-h2o.merge(train_data,stay,by = "Stay_In_Current_City_Years",all.x = T)
train_data<-h2o.merge(train_data,user,by = "User_ID",all.x = T)
train_data<-h2o.merge(train_data,product,by = "Product_ID",all.x = T)

#Combine with test data
test_data<-h2o.merge(test_data,ms,by = "Marital_Status",all.x = T)
test_data<-h2o.merge(test_data,gen,by = "Gender",all.x = T)
test_data<-h2o.merge(test_data,occ,by = "Occupation",all.x = T)
test_data<-h2o.merge(test_data,city,by = "City_Category",all.x = T)
test_data<-h2o.merge(test_data,age,by = "Age",all.x = T)
test_data<-h2o.merge(test_data,stay,by = "Stay_In_Current_City_Years",all.x = T)
test_data<-h2o.merge(test_data,user,by = "User_ID",all.x = T)
test_data<-h2o.merge(test_data,product,by = "Product_ID",all.x = T)

#Removing unnecessary objects
rm(ms)
rm(gen)
rm(occ)
rm(city)
rm(age)
rm(stay)
rm(user)
rm(product)

#________________________________________________________________________________

#Create Interaction columns for categorical variables for train data
Interactions<-h2o.interaction(data=train_data,factors = c("User_ID","Gender","Age","Product_ID","Marital_Status","Occupation","City_Category","Stay_In_Current_City_Years"),pairwise = T,max_factors = 10000,min_occurrence = 1)
head(Interactions,10)
#Combine Interactions with original data
train_data<-h2o.cbind(train_data,Interactions) #Ordinary cbind would not work
#head(all_data,10)
#Remove Interactions
h2o.rm(Interactions)

#Create Interaction columns for categorical variables for test data
Interactions<-h2o.interaction(data=test_data,factors = c("User_ID","Gender","Age","Product_ID","Marital_Status","Occupation","City_Category","Stay_In_Current_City_Years"),pairwise = T,max_factors = 10000,min_occurrence = 1)
head(Interactions,10)
#Combine Interactions with original data
test_data<-h2o.cbind(test_data,Interactions) #Ordinary cbind would not work
#head(all_data,10)
#Remove Interactions
h2o.rm(Interactions)

#________________________________________________________________________________
#One hot encoding and variable clean up
#Read data
train_df<-fread("C:\\Users\\Deepak Kumar Sisodia\\Desktop\\R Project 2\\train.csv",sep = ',')
test_df<-fread("C:\\Users\\Deepak Kumar Sisodia\\Desktop\\R Project 2\\test.csv",sep = ',')

#Converting to correct variable types for train_df
train_df$User_ID<-as.factor(train_df$User_ID)
train_df$Product_ID<-as.factor(train_df$Product_ID)
train_df$Gender<-as.factor(train_df$Gender)
train_df$Age<-as.factor(train_df$Age)
train_df$Occupation<-as.factor(train_df$Occupation) 
train_df$City_Category<-as.factor(train_df$City_Category)
train_df$Stay_In_Current_City_Years<-as.factor(train_df$Stay_In_Current_City_Years)
train_df$Marital_Status<-as.factor(train_df$Marital_Status)

#Converting to correct variable types for test_df
test_df$User_ID<-as.factor(test_df$User_ID)
test_df$Product_ID<-as.factor(test_df$Product_ID)
test_df$Gender<-as.factor(test_df$Gender)
test_df$Age<-as.factor(test_df$Age)
test_df$Occupation<-as.factor(test_df$Occupation) 
test_df$City_Category<-as.factor(test_df$City_Category)
test_df$Stay_In_Current_City_Years<-as.factor(test_df$Stay_In_Current_City_Years)
test_df$Marital_Status<-as.factor(test_df$Marital_Status)

#One hot encoding
library(dummies)
train_df<-dummy.data.frame(train_df,names = c("Gender","Age","Occupation","City_Category","Stay_In_Current_City_Years","Marital_Status"))
test_df<-dummy.data.frame(test_df,names = c("Gender","Age","Occupation","City_Category","Stay_In_Current_City_Years","Marital_Status"))

#Imputing missing values for Product_Category_2 and Product_Category_3 and 
#Creating a flag to denote imputation
train_df<-mutate(train_df,missingProd2=ifelse(is.na(Product_Category_2),1,0),
                 missingProd3=ifelse(is.na(Product_Category_3),1,0),
                 numProds=ifelse(missingProd2==1&missingProd3==1,1
                                 ,ifelse(missingProd2==0&missingProd3==0,3,2)))%>%
    mutate(Product_Category_2_Imp=ifelse(is.na(Product_Category_2),9999,Product_Category_2),
           Product_Category_3_Imp=ifelse(is.na(Product_Category_3),9999,Product_Category_3)
    ,Product_Category_2_NS=ifelse(is.na(Product_Category_2),"NS",Product_Category_2),
    Product_Category_3_NS=ifelse(is.na(Product_Category_3),"NS",Product_Category_3))%>%select(-User_ID,-Product_ID,-Product_Category_1,-Product_Category_2,-Product_Category_3,-Purchase)

test_df<-mutate(test_df,missingProd2=ifelse(is.na(Product_Category_2),1,0),
                 missingProd3=ifelse(is.na(Product_Category_3),1,0),
                 numProds=ifelse(missingProd2==1&missingProd3==1,1
                                 ,ifelse(missingProd2==0&missingProd3==0,3,2)))%>%
    mutate(Product_Category_2_Imp=ifelse(is.na(Product_Category_2),9999,Product_Category_2),
           Product_Category_3_Imp=ifelse(is.na(Product_Category_3),9999,Product_Category_3)
    ,Product_Category_2_NS=ifelse(is.na(Product_Category_2),"NS",Product_Category_2),
    Product_Category_3_NS=ifelse(is.na(Product_Category_3),"NS",Product_Category_3))%>%select(-User_ID,-Product_ID,-Product_Category_1,-Product_Category_2,-Product_Category_3)


#Uploading the data frames to h2o cloud as a h20 frame
train_df<-as.h2o(train_df)
test_df<-as.h2o(test_df)



#Combing the two h2o frames
train_all<-h2o.cbind(train_data,train_df)
test_all<-h2o.cbind(test_data,test_df)


#Splitting train data for modeling:
model_split <- h2o.splitFrame(data = train_all, ratios = 0.75)
#This results in a list containing the frames in the specified proportions
train<-model_split[[1]]
test<-model_split[[2]]

#Removing unnecessary variables
h2o.rm(train_data)
h2o.rm(test_data)
h2o.rm(train_df)
h2o.rm(test_df)
h2o.rm(model_split)
h2o.rm(train_all)
h2o.rm(test_all)

#Random Forest Implementation in H2O

#Getting dimesions and names of train data 
nrow(train)
ncol(train)
names(train)

#Setting variable indices for response
y = 12
#Setting variable indices for predictors
x = c(4:11,16:25,28:31,39,41:43,46:48,54:55,57)
length(x)

#Estimating execution time

stime_1k = Sys.time()
#Creating the model
rand.for.1k = h2o.randomForest(x = x,y = y,training_frame = train,
                               model_id = 'rand.for.1k',seed = 14,mtries = 5,ntrees = 1000)
etime_1k = Sys.time()

stime_1k
etime_1k


#Check the model performance on the training data
h2o.performance(model = rand.for.1k)

#Evaluate the model performance on test/validation data
rand.for.test.perf = h2o.performance(model = rand.for.1k,newdata = test)
rand.for.test.perf

#Grid Search and Model Selection

#1. Defining the tuning parameter - mtries
randfor_pars = list(mtries=seq(2,12,1),ntrees=c(400),max_depth=c(10))
#Capture the start time
time_start_gr = Sys.time()
#2. Using the h2o.grid function, specify the hyper-parameters to be tuned
randfor_grid1 = h2o.grid(algorithm = "randomForest",grid_id = "randfor_grid1",x=x,y=y,training_frame=train,
                         validation_frame=test,seed=84,hyper_params = randfor_pars)
#Capture the end time
time_end_gr = Sys.time()
#3. Sort the resulting models based on the lowest value of RMSE
randfor_perf_grid = h2o.getGrid(grid_id = "randfor_grid1",sort_by = 'RMSE',decreasing = FALSE)
#Print all the models developed
print(randfor_perf_grid)
#4. Select the best model from the various models developed by the grid search
randfor_best_id = randfor_perf_grid@model_ids[[1]]
h20.randfor_best = h2o.getModel(randfor_best_id)
h2o.performance(h20.randfor_best)
#Obtain the Variable importance plot
h2o.varimp_plot(h20.randfor_best)

#Removing variables
rm(randfor_pars)
rm(time_end_gr)
rm(x)
rm(y)

#________________________________________________________________________________

#Deep Learning
# Specify Dependent and Independent Variables
dep.var="Purchase"
indep.var=c("Occupation","City_Category","Marital_Status1","Product_Category_1","Product_Category_2_NS",
            "Product_Category_3_NS","Stay_numeric","Age_numeric","Gender_numeric","Count_User","Count_Product")


#Building DL model  with 2 hidden layers and each containing 50 neurons
train.h2o<-train
valid.h2o<-test

#Building the model
DL <- h2o.deeplearning(x = indep.var,
                       y = dep.var,
                       training_frame = train.h2o,
                       model_id = "dl_fit1",
                       hidden = c(50,50),
                       seed = 1000)

#Check the model performance on validation dataset
DL.perf1 = h2o.performance(model = DL,newdata = valid.h2o)
#Compute the RMSE 
h2o.rmse(DL.perf1)



#________________________________________________________________________________

#Model built with 2 hidden layers and with stopping criteria in terms of mean squared 
#error. 3 fold cross validation is used to build the model.
DL_2 <- h2o.deeplearning(x = indep.var,
                         y = dep.var,
                         training_frame=train.h2o,
                         validation_frame=valid.h2o, 
                         model_id = "DL_3",
                         epochs = 200,
                         hidden = c(20,20),
                         nfolds = 3,                            
                         score_interval = 1,                    
                         stopping_rounds = 5,                   
                         stopping_metric = "MSE", 
                         stopping_tolerance = 1e-3,             
                         seed = 5555)

#Plot that shows the change in RMSE for change in the value of epochs. 
#different lines are plotted for trainning and validation 
plot(DL_1, 
     timestep = "epochs", 
     metric = "rmse")

#________________________________________________________________________________
#Grid of hyper parameters to search for best combination
hyper_params <- list(hidden=list(c(32,32,32),c(64,64,64)),
                     input_dropout_ratio=c(0,0.05),
                     rate=c(0.01,0.02)
                    )

#Using RandomDiscrete as strategy
search_criteria2 <- list(strategy = "RandomDiscrete", 
                         max_models = 36)

grid <- h2o.grid(algorithm="deeplearning", grid_id="dl_grid", 
                 training_frame=train.h2o, validation_frame=valid.h2o, 
                 x = indep.var, y = dep.var, epochs=50,
                 stopping_metric="MSE", stopping_tolerance=1e-2,        
                 adaptive_rate=F, momentum_start=0.5, momentum_stable=0.9, 
                 momentum_ramp=1e7, l1=1e-5, l2=1e-5, 
                 activation = "Rectifier",
                 hyper_params = hyper_params,
                 search_criteria = search_criteria2
                )

grid <- h2o.getGrid("dl_grid",sort_by="rmse",decreasing=FALSE)

#________________________________________________________________________________

#Selected model from the Grid Search
DL_final <- h2o.deeplearning(x = indep.var,
                       y = dep.var,
                       training_frame=train.h2o,
                       validation_frame=valid.h2o, 
                       model_id = "dl_final",
                       hidden = c(50,50),
                       epochs = 50,
                       activation = "Rectifier",
                       input_dropout_ratio = 0.05,
                       rate = 0.01,
                       seed = 1000)

DL.pred.final = h2o.performance(model = DL_final,newdata = valid.h2o)
h2o.rmse(DL.pred.final)

#________________________________________________________________________________

#Prediction
predictions.DL <- as.data.frame(h2o.predict(DL_final, test.h2o))
sub_gbm <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predictions.DL$predict)
write.csv(sub_gbm, file = "E:\\R Ram Gopal 2\\Pr 2\\DL.csv", row.names = F)

#Removing unnecessary variables
h2o.rm(train.h2o)
h2o.rm(valid.h2o)
#________________________________________________________________________________

##########GBM Part 1################

valid<-test

# Specify Dependent and Independent Variables
dep.var="Purchase"
indep.var=c("Occupation","City_Category","Marital_Status1","Product_Category_1","Product_Category_2_NS",
            "Product_Category_3_NS","Stay_numeric","Age_numeric","Gender_numeric","Count_User","Count_Product")

# First gbm model with default parameters(ntrees = 50 by default)

gbm.model.1 <- h2o.gbm(y=dep.var, x=indep.var, training_frame = train,
                      model_id = "gbm.model.1",
                      seed = 1000)

# Second gbm model with number of trees = 500 parameters

gbm.model.2 <- h2o.gbm(y=dep.var, x=indep.var, training_frame = train,
                      model_id = "gbm.model.2",
                      ntrees = 500,
                      seed = 1000)

# Use early stopping functionality to find optimum number to trees to avoid overfitting.
# `score_tree_interval = 3` will score the model after every 3 trees. 
# Model will stop training after there have been 4 scoring intervals 
# where the MSE has not increased more than 0.5

gbm.model.3 <- h2o.gbm(y=dep.var, x=indep.var, training_frame = train,
                    model_id = "gbm.model.3",
                    validation_frame = valid,
                    ntrees = 500,
                    score_tree_interval = 3,
                    stopping_rounds = 4,
                    stopping_metric = "MSE",
                    stopping_tolerance = 0.5,
                    seed = 1000)

# Check metrics(RMSE) for both the above models
gbm.performance.1 = h2o.performance(model = gbm.model.1,newdata = valid)
gbm.performance.2 = h2o.performance(model = gbm.model.2,newdata = valid)
gbm.performance.3 = print(gbm.model.3)

##############################GBM Part 2################

# Specify a grid of parameters which will result in multiple models with all possible combinations of parameters
# Parameters grid-- Total 2*3*2*3 i.e. 36 models wil be created 
gbm_params.1 <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(3, 6, 9),
                    sample_rate = c(0.7, 1.0),
                    col_sample_rate = c(0.3, 0.5, 1.0))

# Give the grid params as input to the model
# Each model's performance would be evaluated on validation data and final metric values would be
# calculated on the validation data
# Specified optimum number of trees = 100 from above analysis

gbm_grid.1 <- h2o.grid("gbm", x = indep.var, y = dep.var,
                      grid_id = "gbm_grid.1",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1000,
                      hyper_params = gbm_params.1)


# Get all model performance metrics in increasing order of the RMSE
gbm_gridperf.1 <- h2o.getGrid(grid_id = "gbm_grid.1", 
                             sort_by = "rmse", 
                             decreasing = F)
print(gbm_gridperf.1)


################################ GBM part 3 ################################

# Now we will specify fixed time (say 180 secs) duration within which several models will be trained
# We can specify the parameter ranges. Parameters values would be picked randomly from these ranges.

gbm_params.2 <- list(learn_rate = seq(0.1, 0.3, 0.01),  #updated
                   max_depth = seq(5, 10, 1), #updated
                   sample_rate = seq(0.9, 1.0, 0.05),  #updated
                   col_sample_rate = seq(0.1, 1.0, 0.1))

# Specify search criteria of 180 secs
search_criteria <- list(strategy = "RandomDiscrete", 
                        max_runtime_secs = 180)  #updated


gbm_grid.2 <- h2o.grid("gbm", x = indep.var, y = dep.var,
                     grid_id = "gbm_grid.2",
                     training_frame = train,
                     validation_frame = valid,
                     ntrees = 100,
                     seed = 1000,
                     hyper_params = gbm_params.2,
                     search_criteria = search_criteria)

# Get all model in increasing order of RMSE values
gbm_gridperf.2 <- h2o.getGrid(grid_id = "gbm_grid.2", 
                            sort_by = "rmse", 
                            decreasing = F)
print(gbm_gridperf.2)

# Get best model from the list of all tried models
best_gbm_model <- gbm_gridperf.2@model_ids[[1]]
best_gbm <- h2o.getModel(best_gbm_model)

# Make predictions on test data
predictions.gbm <- as.data.frame(h2o.predict(best_gbm, test_all))

#________________________________________________________________________________

####### GLM: ########

#### Model -1 :

# Initializing output variable - Purchase
op <- "Purchase"

# Initializing input variables
ip <- c(51:90,93:95)

system.time(
    linear_model <- h2o.glm( y = op, x = ip, training_frame = train, family = "gaussian")
)

h2o.performance(linear_model)



#### Model -2 :

# Initializing output variable - Purchase
op <- "Purchase"

# Initializing input variables
ip <- c(4,5,7,9:11,93:95)

system.time(
    linear_model2 <- h2o.glm( y = op, x = ip, training_frame = train, family = "gaussian")
)

h2o.performance(linear_model2)




#making predictions
linear_predict <- as.data.frame(h2o.predict(linear_model2, test))
#create a data frame and writing submission file
a = as.vector(test$User_ID)
b = as.vector(test$Product_ID)
c = as.vector(linear_predict$predict)
predicted_purchase <- data.frame(User_ID = a, Product_ID = b, Purchase = c)
