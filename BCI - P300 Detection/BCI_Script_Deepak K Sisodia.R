library(e1071)
library(h2o)
library(caTools)
library(gbm)
library(dismo)
library(R.matlab)

################################################################################
#read data from matlab file data.mat and create a csv file 

pathname = file.path("./BCI", "data.mat")
data1 = readMat(pathname)

itr=(dim((data1$x)))[3]
a = data.frame()

for(i in 1:itr)
{
  b = c()
  for(j in 1:32)
  {
    b = c(b,data1$x[,,i][j,])
  }
  a = rbind(a,b)
}

colnames(a)=paste("x",seq(1,4096),sep="")
a$y=as.vector(data1$y)
head(a,1)
write.csv(a,"matcsv.csv",row.names = F)
###############################################################
# Read csv file
data=read.csv("matcsv.csv")
data$y=as.factor(data$y)

split = sample.split(data$y,SplitRatio = 0.8)
train1 = subset(data,split == T)
valid1 = subset(data,split == F)
table(valid1$y)
dim(train1)
##H2o Frame##########
localh2o = h2o.init(nthreads = -1, max_mem_size = "4G")
train_h2o = as.h2o(train1)
valid_h2o = as.h2o(valid1)

dim(train_h2o)
colnames(train_h2o)

################################ Boosting - GBM  #################################
gbm.model.1 = h2o.gbm(y="y",x=colnames(train_h2o)[1:4096],training_frame = train_h2o,
                      model_id = "gbm.model.1",
                      validation_frame = valid_h2o,
                      ntrees = 100,
                      score_tree_interval = 3,
                      stopping_rounds = 4,
                      stopping_metric = "misclassification",
                      seed = 1000)

gbm_param = list(learn_rate = c(0.01,0.1),
                 max_depth = c(3,6,9),
                 sample_rate=c(0.7,1),
                 col_sample_rate = c(0.3,0.5,1))

search_criteria1 = list(strategy = "RandomDiscrete",
                       max_models=10)

gbm_grid.1 = h2o.grid("gbm",x=colnames(data)[1:4096],
                      y="y",
                      grid_id="gbm_grid.1",
                      training_frame = train_h2o,
                      validation_frame = valid_h2o,
                      ntrees = 100,
                      seed=1000,
                      hyper_params = gbm_param,
                      search_criteria = search_criteria1)
gbm_gridperf=h2o.getGrid(grid_id="gbm_grid.1",
                         sort_by="auc",
                         decreasing = T)

best_gbm_model=gbm_gridperf@model_ids[[1]]
h2o.getModel(best_gbm_model)


##########################################################################################

################################ Bagging - Random Forest #################################

rf_fit = h2o.randomForest(x = colnames(data)[1:4096],
                            y = "y",
                            training_frame = train_h2o,
                            model_id = "rf_fit",
                            ntrees = 200,
                            seed = 1000)

rf_perf = h2o.performance(model = rf_fit,newdata = valid_h2o)
h2o.auc(rf_perf)


############################################################################################

h2o.shutdown()







