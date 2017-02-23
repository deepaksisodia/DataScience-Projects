# setwd("C:\\Users\\Deepak Kumar Sisodia\\Desktop\\Data Challenge\\train")
library(FeatureHashing)
library(caTools)
library(data.table)
library(h2o)
library(glmnet)

####### Loading Sampled data file ######################################################################
getwd()
Final.Sample = read.csv("small_1%_data.csv") # New Line added


####### Some summary / Data Exploration ################################################################
str(Final.Sample)
summary(Final.Sample)
head(Final.Sample)

table(Final.Sample$hour)
prop.table(table(Final.Sample$click))

#################################################

######## Convering Data type for this data set ##########################################################

Final.Sample$click=as.factor(Final.Sample$click)
Final.Sample$C1=as.factor(Final.Sample$C1)
Final.Sample$banner_pos=as.factor(Final.Sample$banner_pos)

Final.Sample$device_type=as.factor(Final.Sample$device_type)
Final.Sample$device_conn_type=as.factor(Final.Sample$device_conn_type)
Final.Sample$C14=as.factor(Final.Sample$C14)
Final.Sample$C15=as.factor(Final.Sample$C15)
Final.Sample$C16=as.factor(Final.Sample$C16)
Final.Sample$C17=as.factor(Final.Sample$C17)
Final.Sample$C18=as.factor(Final.Sample$C18)
Final.Sample$C19=as.factor(Final.Sample$C19)
Final.Sample$C20=as.factor(Final.Sample$C20)
Final.Sample$C21=as.factor(Final.Sample$C21)

str(Final.Sample)
View(Final.Sample)

#################################################

########### Creating feature "day of week" #############################################################
startYM=1410
startDH=21
i=2
Final.Sample$day_of_week = 8

for (day in seq(21,30,1))
{
  
  startdaytime = startYM*100 + day
  Final.Sample[(Final.Sample$hour)%/%100 == startdaytime,'day_of_week'] = i
  i=i+1
  i = ifelse(i<=7,i,1)
  
}

# yy=as.character(Final.Sample$hour)
# weekdays(as.Date("14102103",format = "%y%m%d%H"))
prop.table(table(Final.Sample$day_of_week))

####################################################

############ Creating feature "hour of the day" ############################################################
Final.Sample$hour_of_day = Final.Sample$hour%%100
prop.table(table(Final.Sample$hour_of_day))

####################################################

######### Rationale for Day of Week and hour of day feature################################################
opar = par()
par(mfrow=c(1,2))
t1 = tapply(Final.Sample$click, Final.Sample$hour, length)
plot(t1)
lines(t1)
t2 = plot(tapply(Final.Sample[Final.Sample$click == 1,'click'],Final.Sample[Final.Sample$click == 1,'hour'] , length))
lines(t2)
par(opar)

####################################################

######## Splitting data into train and validation #########################################################
sample = sample.split(Final.Sample$click,SplitRatio = 0.3)
Final.Sample.Train = subset(Final.Sample,sample==F) 
nrow(Final.Sample.Train)
Final.Sample.Train[] = lapply(Final.Sample.Train, function(x) if(is.factor(x)) factor(x) else x)

Final.Sample.Test = subset(Final.Sample,sample==T)
nrow(Final.Sample.Test)
Final.Sample.Test[] = lapply(Final.Sample.Test, function(x) if(is.factor(x)) factor(x) else x)

#####################################################

########### funnction to create prob of (Target = 1) given a particular class in a categorical variable in Train dataset ###########3
Target_prop_features_train = function(orig_data,column_names,target_col_name)
{
  
  Merged_df_train = orig_data
  gn_column_level_count = c()
  gn_df_rbind = data.frame()
  
  for (i in column_names)
  {
    
    gn_column = tapply(as.integer(as.character(orig_data[,target_col_name])),orig_data[,i], mean)
    gn_df= data.frame(gn_column,names(gn_column))
    
    colnames(gn_df) = c(paste(i,sep = "","_prop"),i)
    
    Merged_df_train = merge(x=Merged_df_train,y=gn_df,by = i,all.x=T)
    
    gn_column_level_count = c(gn_column_level_count,nrow(gn_df))
    gn_df$feature_name = i
    colnames(gn_df) = c("C1","C2","C3")
    gn_df_rbind = rbind(gn_df_rbind,gn_df)
    
    rm(gn_column)
    rm(gn_df)
  }
  
  return_list_obj = list("L1" = Merged_df_train, "L2" = gn_df_rbind, "L3" = gn_column_level_count)
  return(return_list_obj)
  
}

########## Function to assign data to test observations for newly created features based on training set ############ 
features_value_assign_test = function(orig_data,gn_df_rbind,Features_select)
{
  
  Merged_df_test = orig_data
  for(i in Features_select)
  {
    # temp_gn_column_df = data.frame(gn_df_rbind[seq(j+1,j+k),c(1,2)])
    temp_gn_column_df = data.frame(gn_df_rbind[gn_df_rbind[,3] == i,c(1,2)])
    colnames(temp_gn_column_df) = c(paste(i,sep = "","_prop"),i)
    Merged_df_test = merge(x=Merged_df_test,y=temp_gn_column_df, by = i,all.x=T)
    rm(temp_gn_column_df)
  }
  
  return(Merged_df_test) 
}


####### Columns selected for feature engineering ##########################################################
Features_select = c("site_id","site_domain","app_id","device_id","device_ip","device_model","app_domain","C14","C17","C20")

#####################################################

######### Function Calling ################################################################################
Received_list_obj_train = Target_prop_features_train(Final.Sample.Train, Features_select, "click")
Final.Sample.Train.features = Received_list_obj_train$L1

Final.Sample.Train.features$site_id_prop=as.numeric(Final.Sample.Train.features$site_id_prop)
Final.Sample.Train.features$site_domain_prop=as.numeric(Final.Sample.Train.features$site_domain_prop)
Final.Sample.Train.features$app_id_prop=as.numeric(Final.Sample.Train.features$app_id_prop)
Final.Sample.Train.features$device_id_prop=as.numeric(Final.Sample.Train.features$device_id_prop)
Final.Sample.Train.features$device_ip_prop=as.numeric(Final.Sample.Train.features$device_ip_prop)
Final.Sample.Train.features$device_model_prop=as.numeric(Final.Sample.Train.features$device_model_prop)
Final.Sample.Train.features$day_of_week=as.factor(Final.Sample.Train.features$day_of_week)
Final.Sample.Train.features$hour_of_day=as.factor(Final.Sample.Train.features$hour_of_day)
Final.Sample.Train.features$app_domain_prop=as.numeric(Final.Sample.Train.features$app_domain_prop)
Final.Sample.Train.features$C14_prop=as.numeric(Final.Sample.Train.features$C14_prop)
Final.Sample.Train.features$C17_prop=as.numeric(Final.Sample.Train.features$C17_prop)
Final.Sample.Train.features$C20_prop=as.numeric(Final.Sample.Train.features$C20_prop)




Final.Sample.Test.features = features_value_assign_test(Final.Sample.Test, Received_list_obj_train$L2, Features_select)
Final.Sample.Test.features$day_of_week=as.factor(Final.Sample.Test.features$day_of_week)
Final.Sample.Test.features$hour_of_day=as.factor(Final.Sample.Test.features$hour_of_day)
Final.Sample.Test.features$C20_prop=as.numeric(Final.Sample.Test.features$C20_prop)


str(Final.Sample.Train.features)
str(Final.Sample.Test.features)


colnames(Final.Sample.Train.features)==colnames(Final.Sample.Test.features)




##################### Treating NA's of test dataset for new features ##############################
colnames(Final.Sample.Test.features)
na.features = c("site_id_prop","site_domain_prop","app_id_prop","device_id_prop","device_ip_prop","device_model_prop"
                ,"app_domain_prop","C14_prop","C17_prop","C20_prop")
mean_clicks = mean(as.numeric(as.character(Final.Sample.Test.features$click)))
for(i in na.features)
{
  Final.Sample.Test.features[is.na(Final.Sample.Test.features[,i]),i] = mean_clicks
}
str(Final.Sample.Test.features)
summary(Final.Sample.Test.features)

Final.Sample.Test.features=Final.Sample.Test.features[!Final.Sample.Test.features$C19 %in% c(545),]


##################### Modeling ####################################################################

ind.var=c("site_id_prop","site_domain_prop",
          "app_id_prop","device_id_prop","device_ip_prop","device_model_prop","app_domain_prop","C14_prop",
          "C17_prop","C20_prop","C14","C17","C20","C19","C21","C15","C16","C18","C1","banner_pos","device_type","device_conn_type",
          "hour_of_day","site_id","site_domain","site_category", "app_id","app_domain","app_category", "device_id","device_ip",
          "device_model")

# ,"app_domain","C14","C17","C20","C19","C21","C1","banner_pos","site_category","app_category","device_type","device_conn_type",
# "C15","C16","C18", ,"hour_of_day","site_id_prop","site_domain_prop",
# "app_id_prop","device_id_prop","device_ip_prop","device_model_prop","app_domain_prop","C14_prop",
# "C17_prop","C20_prop"
dep.var="click"
str(Final.Sample.Train.features)

############### Evaluation Metrics function ###############################
MultiLogLoss <- function(act, pred)
{
  eps <- 1e-15
  pred <- pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}

################################################


############## Logistic Regression #####################################################

ss=""
for (item in ind.var){
  ss=ifelse(ss=="",item,paste(ss,item,sep="+"))
}

formula=paste(dep.var,"~",ss)
logit_model=glm(formula,family="binomial",data=Final.Sample.Train.features)

summary(logit_model)

predicted_prob = predict(logit_model,type = "response",newdata = Final.Sample.Test.features)

MultiLogLoss(as.numeric(as.character(Final.Sample.Test.features$click)),predicted_prob)

#####################################################

############# Logistic Regression with Hasing technique #####################################
ind.var.glm=c("device_model", "device_ip", "device_id", "app_id","site_domain","site_id","C1","banner_pos",
              "site_category","app_domain","app_category","device_type","device_conn_type","C14",
              "C15","C16","C17","C18","C19","C20","C21", "day_of_week","hour_of_day")

objTrain_hashed = hashed.model.matrix(~., data=Final.Sample.Train.features[,ind.var.glm], hash.size=2^10, transpose=FALSE)
objTrain_hashed = as(objTrain_hashed, "dgCMatrix")
objTest_hashed = hashed.model.matrix(~., data=Final.Sample.Test.features[,ind.var.glm], hash.size=2^10, transpose=FALSE)
objTest_hashed = as(objTest_hashed, "dgCMatrix")


glmnetModel <- cv.glmnet(objTrain_hashed, Final.Sample.Train.features[,dep.var], 
                         family = "binomial", type.measure = "auc")

glmnetPredict <- predict(glmnetModel, objTest_hashed, s="lambda.min",type="response")
MultiLogLoss(as.numeric(as.character(Final.Sample.Test.features$click)),glmnetPredict[,1])


#################### GBM with Hashing Technique ###################################################
objTrain_hashed1 = hashed.model.matrix(~., data=Final.Sample.Train.features[,ind.var.glm], hash.size=2^8, transpose=FALSE)
dim(objTrain_hashed1)
nd = c()
for(i in seq(1,256))
{
  nd=cbind(nd,objTrain_hashed1[,i])
}
dim(nd)
nd_df = data.frame(nd)
dim(nd_df)
nd_df$click = Final.Sample.Train.features$click

objTest_hashed1 = hashed.model.matrix(~., data=Final.Sample.Test.features[,ind.var.glm], hash.size=2^8, transpose=FALSE)
dim(objTest_hashed1)
ndt = c()
for(i in seq(1,256))
{
  ndt=cbind(ndt,objTest_hashed1[,i])
}
dim(ndt)
ndt_df = data.frame(ndt)
dim(ndt_df)
ndt_df$click = Final.Sample.Test.features$click



localh2o = h2o.init(nthreads = -1, max_mem_size = "4G")
train_h2o = as.h2o(nd_df)
valid_h2o = as.h2o(ndt_df)

dim(train_h2o)
dim(valid_h2o)
inputvars=setdiff(colnames(train_h2o),"click")

gbm.model.1.hash = h2o.gbm(y="click",x=inputvars,training_frame = train_h2o,
                      model_id = "gbm.model.1.hash",
                      validation_frame = valid_h2o,
                      ntrees = 100,
                      score_tree_interval = 3,
                      stopping_rounds = 4,
                      stopping_metric = "logloss",
                      seed = 1000)

summary(gbm.model.1.hash)
#####################################################

############### H20 Frame ################################################################

localh2o = h2o.init(nthreads = -1, max_mem_size = "4G")
train_h2o = as.h2o(nd_df)
valid_h2o = as.h2o(ndt_df)

dim(train_h2o)
colnames(train_h2o)

################ Random Forest -Tree based approach ##############################################

rf_fit = h2o.randomForest(x = ind.var,
                          y = dep.var,
                          training_frame = train_h2o,
                          validation_frame = valid_h2o,
                          model_id = "rf_fit",
                          ntrees = 300,
                          stopping_rounds = 5,
                          score_each_iteration = T,
                          seed = 1000)

rf_fit@model$validation_metrics

######################################################


################################ Boosting - GBM  #################################
gbm.model.1 = h2o.gbm(y=dep.var,x=ind.var,training_frame = train_h2o,
                      model_id = "gbm.model.1",
                      validation_frame = valid_h2o,
                      ntrees = 100,
                      score_tree_interval = 3,
                      stopping_rounds = 4,
                      stopping_metric = "logloss",
                      seed = 1000)


# gbm_param = list(learn_rate = c(0.01,0.005),
#                  max_depth = c(2,3,4,5),
#                  sample_rate=c(0.5,0.6,0.7),
#                  col_sample_rate = c(0.2,0.3,0.5))

gbm_param_fixed = list(learn_rate = 0.01,
                       max_depth = 4,
                       sample_rate=0.7,
                       col_sample_rate = 0.2)

# search_criteria1 = list(strategy = "RandomDiscrete",
#                         max_models=50)


gbm_grid.1 = h2o.grid("gbm",x=ind.var,
                      y=dep.var,
                      grid_id="gbm_grid.3",
                      training_frame = train_h2o,
                      validation_frame = valid_h2o,
                      ntrees = 100,
                      seed=1000,
                      hyper_params = gbm_param_fixed)

summary(gbm_grid.1)
                      

# gbm_gridperf=h2o.getGrid(grid_id="gbm_grid.1",
#                          sort_by="logloss",
#                          decreasing = F)
# 
# best_gbm_model=gbm_gridperf@model_ids[[1]]
# h2o.getModel(best_gbm_model)

#######################################################












