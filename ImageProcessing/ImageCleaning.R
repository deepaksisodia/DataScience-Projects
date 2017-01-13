#Feature Engineering
getLineWidth <- function(imgX)

{
# Computing the row wise mean-pixel value for each pixel row in the image 
vectorImgX = unlist(apply(imgX, 1, mean)) # Apply function will calculate row-wise "mean" pixel brightness for all the 258 rows in imgX matrix. The '1' parameter means apply the mean operation row wise.

# Plotting the mean pixel for each row versus the rows in the image
x = 1:nrow(imgX)    # There are 258 rows in imgX matrix. 
plot(x, vectorImgX, col="black", type="l", xlab = "row", ylab = "mean brightness")


#### CLUSTERING BASED ON AVERAGE ROW BRIGHTNESS TO IDENTIFY THE LOCATION OF LINES OF TEXT:

# Making 2 cluster for brightness values: close to 0 (dark) and close to 1 (bright).
clstr = kmeans(vectorImgX, 2)

#Defining cutoff point as the mean of the highest value in the lower cluster and the lowest value in the lower cluster.
cutoff = ((max(vectorImgX[clstr$cluster == which.min(clstr$centers)]) + min(vectorImgX[clstr$cluster == which.max(clstr$centers)])) / 2)

# Marking a cutoff line on the 'mean brightness' versus row graph.
lines(vectorImgX * 0 + cutoff, col="red")

# Identifying all the rows where there a transition from white pixel (pixel value above the cutoff) to the dark pixel (pixel value below the cutoff):
spikes = c()
for(i in 2:length(vectorImgX))
{
    if(vectorImgX[i] < cutoff & vectorImgX[i-1] > cutoff)
    {
        spikes <- c(spikes, i)
    }
}

# Calculating the mean value of rows after which we expect line of sentence (spike/transition from high pixel value to low pixel value) to occur.
cycleLength = mean(diff(spikes))

# Creating a vector to find the spaces between the lines and initializing it with 0.
spaces = matrix(0, length(spikes) - 1)

# Creating a dummy vector (dummyImgX) to find the pattern of the transition of pixels in subject image for comparison later.
dummyImgX = vectorImgX

# Assigning all values which are greater than cutoff to pixel value of 1 (background pixel / brightest) and which are not making any changes to the values which are lower than cutoff.
for (i in 2:length(spikes))
{
    cycleStarts = spikes[i]                  # In my case, spikes[1] is 15, i.e at row 15 of the pixel matrix, there was a transition from white to black.
    while (vectorImgX[cycleStarts] < cutoff) # While the value of "vectorImgX" (Mean value of the pixels in a given row) is less than that of cutoff.
        cycleStarts = cycleStarts - 1          # If the pixel is less than cutoff (i.e. dark), the value of cycleStarts is reduced by 1 to recheck the condition. WE WANT TO CHECK WHEN WE CAN EXPECT THE VALUE OF PIXEL THAT IS GREATER THAN CUTOFF.
    gapStarts = cycleStarts - 1              # This happens if the above while loop fails. At this point, gapstarts is also assigned a value, which is equal to the new (reduced) value of cycleStarts.
    while (vectorImgX[gapStarts] > cutoff)   # This loop will be executed when that pixel value will be greater than cutoff. At this point this while loop will execute to FIND THE GAP WHERE PIXELS ARE BRIGHT (greater than cutoff).
    {  
        gapStarts = gapStarts - 1              # Value of gapstarts is reduced in every iteration to check the condition (in while loop) if the pixel value at that point is greater than the cutoff.
        dummyImgX[gapStarts]=1                 # At this point, we are assigning the pixel value of dummy vector equal to 1 (for comparison later).
    }
}


plot(x, dummyImgX, col="red", type="l", xlab = "row", ylab = "brightness")

# Initializing a new matrix to take the store the cleaned image. Initializing it to the original image and later compared it to the imputed value of background pixel.
imgCleaned = imgX
# Converting the new matrix into vector
matCln = matrix(imgCleaned, nrow(imgX) * ncol(imgX), 1)
#Imputing the detected gaps with a pixel value of 1.
matCln = ifelse(pmax(matCln,dummyImgX)>0.99,1,matCln)

}

#________________________________________________________________________________________
#Edge detection and Morphological techniques
pixelDilation <-function(imageinp){
  imagecopy=imageinp
  r=dim(imageinp)[1]
  c=dim(imageinp)[2]
  
  for(i in (1:r)){
    #print(paste("i",i))
    for(j in (1:c)){
      #print(paste("j",j))
      
      if (imageinp[i,j]==1){
        
        if(i!=r & j!=c){
          sub_image=rbind(imageinp[i-1,(j-1):(j+1)],imageinp[i,(j-1):(j+1)],imageinp[i+1,(j-1):(j+1)])
        }
        else{
          if(i==r & j==c)
          {
            sub_image=rbind(imageinp[i-1,(j-1):(j)],imageinp[i,(j-1):(j)])
          }
          else if(j==c)
          {
            sub_image=rbind(imageinp[i-1,(j-1):(j)],imageinp[i,(j-1):(j)],imageinp[i+1,(j-1):(j)])
          }
          else{
            sub_image=rbind(imageinp[i-1,(j-1):(j+1)],imageinp[i,(j-1):(j+1)])
          }
        }
        
        mask=matrix(1,nrow(sub_image),ncol(sub_image))
        xor.output=xor(sub_image,mask)
        
        if(sum(xor.output)>0){
          imagecopy[i,j]=0
        }
        
      }#close if pixel == 1
      
    }
  }
  return(imagecopy)
}
pixelErosion <-
function(imageinp){
  imagecopy=imageinp
  r=dim(imageinp)[1]
  c=dim(imageinp)[2]
  
  for(i in 1:r){
    
    for(j in 1:c){
      
      if (imageinp[i,j]==0){
        
        if(i!=r & j!=c){
          sub_image=rbind(imageinp[i-1,(j-1):(j+1)],imageinp[i,(j-1):(j+1)],imageinp[i+1,(j-1):(j+1)])
        }
        else{
          if(i==r & j==c)
          {
            sub_image=rbind(imageinp[i-1,(j-1):(j)],imageinp[i,(j-1):(j)])
          }
          else if(j==c)
          {
            sub_image=rbind(imageinp[i-1,(j-1):(j)],imageinp[i,(j-1):(j)],imageinp[i+1,(j-1):(j)])
          }
          else{
            sub_image=rbind(imageinp[i-1,(j-1):(j+1)],imageinp[i,(j-1):(j+1)])
          }
        }
        
        mask=matrix(0,nrow(sub_image),ncol(sub_image))
        xor.output=xor(sub_image,mask)
        
        if(sum(xor.output)>0){
          imagecopy[i,j]=1
        }
        
      }#close of pixel ==0
    }
  }
  
  return(imagecopy)
}

EdgeDetectionandImageMorphology = function(image.matrix){
    
    imagedata.inputimage=imagedata(image.matrix*255)
    
    imagedata.edgedetected = imgCanny(imagedata.inputimage, 0.7)
    
    matrix.edgedetected=imagedata.edgedetected[1:nrow(imagedata.edgedetected),1:ncol(imagedata.edgedetected)]
    matrix.edgedetected.binary=matrix.edgedetected/255
    feature2= matrix(matrix.edgedetected.binary,ncol = 1)
    
    dilatedimage=pixelDilation(matrix.edgedetected.binary)
    erodedimage=pixelErosion(dilatedimage)
    feature3= matrix(erodedimage,ncol = 1)
    
    erodedimage2=pixelErosion(erodedimage)
    dilatedimage2=pixelDilation(erodedimage2)
    feature4= matrix(dilatedimage2,ncol = 1)
    
    feature.matrix=cbind(feature2,feature3,feature4)
    colnames(feature.matrix)=c("Edge","Edge+D+E","Edge+D+E+E+D")
    
    return(feature.matrix)
}

#__________________________________________________________________________________________

#Convert image into a single column matrix
img2vec <-
function(img)
{
    return (matrix(img, nrow(img) * ncol(img), 1))
}

#_____________________________________________________________________________________________
#Median Filtering
median_filter <-function(img,k=13)
{
  
  n = floor(k/2)  
  
  #Modify the input image matrix by padding 0s outside the input image matrix
  #to make it size having an additional n rows and n columns
  img_modify = matrix(0,nrow(img)+2*n,ncol(img)+2*n)
  
  #Create an empty output matrix of the same size as the input image 
  
  background = matrix(0,nrow(img),ncol(img))
  
  #Copy the Original matrix to the zero/padded matrix
  
  row_seq = seq(nrow(img))
  col_seq = seq(ncol(img))
  
  for (x in row_seq)
  {
    for (y in col_seq)
    {
      img_modify[x+n,y+n] = img[x,y]
    }
  }
  
  #The new modified image matrix has additional 0s padded in the additional n rows and n columns
  #surrounding the input image matrix
  #View(img_modify)
  
  #Store a k-k neighbour values in the array
  #Sort and Find the middle element
  
  for (i in seq(nrow(img_modify)-k))
  {
    for (j in seq(ncol(img_modify)-k))
    {
      window = matrix(0,k*k,1)
      c = 1
      for (x in seq(k))
      {
        for (y in seq(k))
        {
          window[c] = img_modify[i+x-1,j+y-1]
          c = c + 1
        }
      }
      
      med = sort(window)
      
      background[i,j] = med[((k*k)+1)/2]
    }
  }
  
  #the cleaned image can be obtained by substracting the original image from the background
  
  foreground = img - background
  
  #In this case, we know that the writing is always darker than the background, 
  #so, our foreground should only show pixels that are darker than background
  foreground[foreground > 0] = 0
  
  #Normalizing the final results (pixels) to lie between 0-1
  
  m1 = min(foreground)
  m2 = max(foreground)
  
  foreground = (foreground - m1) / (m2 - m1)
  
  return(foreground)
  
}
EdgeDetectionandImageMorphology <-
function(image.matrix){
  
  imagedata.inputimage=imagedata(image.matrix*255)
  
  imagedata.edgedetected = imgCanny(imagedata.inputimage, 0.7)
  
  matrix.edgedetected=imagedata.edgedetected[1:nrow(imagedata.edgedetected),1:ncol(imagedata.edgedetected)]
  matrix.edgedetected.binary=matrix.edgedetected/255
  feature2= matrix(matrix.edgedetected.binary,ncol = 1)
  
  dilatedimage=pixelDilation(matrix.edgedetected.binary)
  erodedimage=pixelErosion(dilatedimage)
  feature3= matrix(erodedimage,ncol = 1)
  
  erodedimage2=pixelErosion(erodedimage)
  dilatedimage2=pixelDilation(erodedimage2)
  feature4= matrix(dilatedimage2,ncol = 1)
  
  feature.matrix=cbind(feature2,feature3,feature4)
  colnames(feature.matrix)=c("Edge","Edge+D+E","Edge+D+E+E+D")
  
  return(feature.matrix)
}

#___________________________________________________________________________________________

#Adaptive Thresholding
Adaptive_thresholding <-function(image)
{
  
  
  # function to convert image to matrix
  image2Matrix = function(Img)
  {
    m1 = t(matrix(Img, nrow(Img), ncol(Img)))
    return(m1)
  }
  
  #load the image
  load_image = readImage(image)

  #Create images with different height and widht 
  threshold_image_2 = thresh(load_image, 2, 2)
  threshold_image_7 = thresh(load_image, 7, 7)
  threshold_image_9 = thresh(load_image, 9, 9)
  threshold_image_12 = thresh(load_image, 12, 12)
  threshold_image_15 = thresh(load_image, 15, 15)
  
  # combine the thresholded images
  combine = cbind(img2vec(image2Matrix(threshold_image_2)),
                img2vec(image2Matrix(threshold_image_7)),
                img2vec(image2Matrix(threshold_image_9)),
                img2vec(image2Matrix(threshold_image_12)),
                img2vec(image2Matrix(threshold_image_15))
                )
  #get the maximum pixel brightness among different images
  final_image = apply(combine, 1, max)
  #form a final matrix
  image_matrix = matrix(final_image, nrow(img), ncol(img))
  #return the image
  return(image_matrix)
  
}


