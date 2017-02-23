
################## Loading required libraries #################################
library(caTools)
library(data.table)

################## Sampling ###################################################
train.small = read.csv(file="train.csv",nrows = 5,skip=0)
column.names=colnames(train.small)
rm(train.small)

startYM=1410
startDH=2100
Final.Sample=data.frame()
number.of.rows=6000000
offset=1
percent_sample = 0.01 # Just Sampling 1% of the data

for (day in seq(startDH,3000,100))
  {
  
  orig_train = fread("train.csv",nrows = number.of.rows,skip=offset,header=F)
  colnames(orig_train)=column.names
  
  hr=seq(day,by = 1,length.out = 24)
  startdaytime=hr+ 1410*10000
  
  
  dayN=orig_train[orig_train$hour %in% startdaytime,]
  offset=offset+nrow(dayN)
  
  Final.DayN.Sample=data.frame()
  
  for (t in startdaytime)
    {
    
    dayN.hourN=dayN[dayN$hour==t,]
    split=sample.split(dayN.hourN$click,SplitRatio = percent_sample)
    sample_dayN_hrN=subset(dayN.hourN,split==TRUE)
    
    Final.DayN.Sample=rbind(Final.DayN.Sample,sample_dayN_hrN)
  }
  
  Final.Sample=rbind(Final.Sample,Final.DayN.Sample)
  
  rm(sample_dayN_hrN)
  rm(Final.DayN.Sample)
  rm(dayN)
  rm(orig_train)
  rm(split)
  
}

write.csv(x = Final.Sample, file = "small_1%_data.csv")