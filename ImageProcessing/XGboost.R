#Clearing all objects
rm(list = ls())

#Setting libraries 
library(png)
library(raster)
library(xgboost)
library(biOps)
library(EBImage)
#________________________________________________________________________________________________________________

#Estimating the time taken
#start_time<-Sys.time()
#Get the input images

#Set folder where the images are located
train_folder<-"C:/Users/chars/Desktop/R/Project/R_Project/Data/train_sample"
target_folder<-"C:/Users/chars/Desktop/R/Project/R_Project/Data/clean_sample"
#test_folder<-"C:/Users/chars/Desktop/R/Project/R_Project/Data/clean_sample/test"

#Obtain a list of all the images in the folder
filenames = list.files(train_folder, pattern="*.png")

#_________________________________________________________________________________________________________________

#Load functions from the previous image cleaning techniques (Present in project folder)
source(file = "ImageCleaning.R")
#_________________________________________________________________________________________________________________

#Creating a dataframe that consists of the cleaned images from the previousimage cleaning techniques.
#These images are converted into a single column matrix and used as predictors for the XGBoost matrix

#Create am empty data frame. The clean images would be appended to this data frame
imgPred<-data.frame()

#Get image cleaning features for each image in the data set

for(file in filenames){
    img = readPNG(file.path(train_folder,file)) #Load image
    clean_img<-readPNG(file.path(target_folder,file)) #Load cleaned image
    x1<-img2vec(img) #Convert image into a single column matrix. This would be the first predictor
    x2<-getLineWidth(img) #Obtain the image after performing featurization to extract line width. 
                          #The output of the function is converted into a vector. This would be the second predictor
    x3<-img2vec(median_filter(img,9)) #Obtain the image after performing median filtering. 
                                    #The result is converted into a single column matrix. This would be the third predictor
  #X4 and x5 have been removed from the data as they were detrimental to the model performance 
    #x4<-EdgeDetectionandImageMorphology(img)
    #x5<-img2vec(Adaptive_thresholding(file.path(train_folder,filenames[1])))
    y<-img2vec(clean_img)  #Convert the cleaned image into a single column matrix
                           #This would be the third predictor
    
    imgPred<-rbind(imgPred,cbind(x1,x2,x3,y)) #Combine all the predictors into a dataframe
    
}


#Gving column names to the dataframe
colnames(imgPred)<-c("Original","Featurization","Median_Filtering","Target")

#To train the model, we are selecting 250000 rows from the data frame
randSample<-sample(nrow(imgPred),250000)

#Convert the data into a dense matrix suitable to be the input for the xgboost model.
#The data is the matrix form of the first three columns of the dataframe and the label is the last column of the dataframe
train<-xgb.DMatrix(as.matrix(imgPred[randSample,-4]),label=imgPred[randSample,4])

#Running a cross validation on the dataset to find the optimum number of rounds to run the model. A maximum of 500 rounds
#is tried. The round with the minimum RMSE is taken as the number of rounds parameter for the model
cv_results = xgb.cv(data = train, nthread = 8, eval_metric = "rmse", nrounds = 500,showsd = T, early.stop.round = 50, nfold = 5,  print.every.n = 10)

#Getting the minimum number of iterations required to get the lowest testing data rmse
min_error_run = which.min(cv_results[, test.rmse.mean]) 


#Creating the xgboost model
xgb<-xgboost(data=train,nthread=2,nrounds = min_error_run) 

# Measuring execution time
# end.time <- Sys.time()
# time.taken <- end.time - start_time
# time.taken

#_________________________________________________________________________________________________________________
#Testing the model with a sample image
#Reading image
pred_Img<-readPNG(file.path(train_folder,filenames[2])) #This can be replaced by other images

#Getting all the predictor values for the image using the image cleaning functions
x1<-img2vec(pred_Img)
x2<-getLineWidth(pred_Img)
x3<-img2vec(median_filter(pred_Img,9))
# x5<-EdgeDetectionandImageMorphology(pred_Img)
# x6<-Adaptive_thresholding(file.path(train_folder,filenames[2]))


#Combining all the predictor values
pred_data<-cbind(x1,x2,x3)
#Naming the columns of the test data
colnames(pred_data)<-c("Original","Featurization","Median Filtering")

#Predicting the target values for the test data using the xgb model
pred<-predict(xgb,pred_data)
#Taking care of values outside [0,1]
pred[pred<0]<-0
pred[pred>1]<-1
#Converting vector back into a matrix of the same dimensions as the original image
clean_img<-matrix(pred,nrow(pred_Img),ncol(pred_Img))
plot(raster(clean_img),col=gray.colors(256,start=0,end=1,gamma=.85,alpha = NULL))
plot(raster(pred_Img),col=gray.colors(256,start=0,end=1,gamma=0.85,alpha = NULL))


#Estimating feature importance 
imp<-xgb.importance(feature_names = colnames(pred_data),model = xgb)
xgb.plot.importance(imp)


#Writing the cleaned image to file
writePNG(clean_img,"clean.png")