# set universal parameter
options(warn=-1)

# import H2O library
library(h2o)
localH2O =  h2o.init(nthreads = -1, port = 54321, max_mem_size = '6G', startH2O = TRUE)

# Students: Use the "absolute" path to the datasets on your machine (important)
labeled.frame <- h2o.importFile(path = '/Users/24653/FIT5201/ass2/Dataset\ and\ Sample\ code/Task2C_labeled.csv' ,sep=',') 
unlabeled.frame <- h2o.importFile(path = '/Users/24653/FIT5201/ass2/Dataset\ and\ Sample\ code//Task2C_unlabeled.csv' ,sep=',') 
test.frame <- h2o.importFile(path = '/Users/24653/FIT5201/ass2/Dataset\ and\ Sample\ code/Task2C_test.csv' ,sep=',') 

labeled.frame[,1] <- as.factor(labeled.frame$label)
unlabeled.frame[,1] <- NA
train.frame <- h2o.rbind(labeled.frame[,-1], unlabeled.frame[,-1])
test.frame[,1] <- as.factor(test.frame$label)


error.rate <- function(Y1, T1){
  if (nrow(Y1)!=nrow(T1)){
    stop('error.rate: size of true lables and predicted labels mismatch')
  }
  return (sum(T1!=Y1)/nrow(T1))
}

reconstruction.train.error <- matrix(NA, nrow=20, ncol=2)
classification.labeled.error <- matrix(NA, nrow=20, ncol=2)

reconstruction.test.error <- matrix(NA, nrow=20, ncol=2)
classification.test.error <- matrix(NA, nrow=20, ncol=2)

# Initial index
index <- 1
for (k in seq(20, 400, 20)){
  # Students: need to write up code here
  ae.model = h2o.deeplearning(    
    x = 2:ncol(train.frame), # select all pixels
    training_frame = train.frame, # specify the frame (imported file)    
    hidden = c(k), # number of layers and their units
    epochs = 50, # maximum number of epoches  
    activation = 'Tanh', # activation function 
    autoencoder = TRUE  # is it an autoencoder? Yes!
)
  # record reconstruction error
  reconstruction.train.error[index,1] <- k
  reconstruction.train.error[index,2] <- mean(h2o.anomaly(ae.model,train.frame))  
  index <-index + 1
    
}

library(ggplot2)

# transform reconstruction error matrix to data frame 
error.df  <- as.data.frame(reconstruction.train.error)
names(error.df) <- c('k','error')
ggplot(data = error.df, aes(x=k,y=error)) + 
  geom_line() + 
  xlab('number of neck neurons') +
  ylab('reconstruction train error')



# Initial index
index <- 1
for (k in seq(20, 400, 20)){
  # Students: need to write up code here
  NN.model <- h2o.deeplearning(    
  x = 2:ncol(labeled.frame), 
  y = 1,
  training_frame = labeled.frame, # specify the frame (imported file)    
  hidden = c(k,k,k), # number of layers and their units
  epochs = 50, # maximum number of epoches  
  activation = 'Tanh', # activation function 
  autoencoder = FALSE, # is it an autoencoder? Yes!
  l2 = 0.1
)

  # record reconstruction error
  classification.test.error[index,1] <- k
  labeled.predict <- h2o.predict(NN.model, labeled.frame)$predict
  classification.test.error[index,2] <-error.rate(labeled.frame$label, labeled.predict) 
  index <- index + 1    
}

# transform classification error matrix to data frame 
error.df.cl  <- as.data.frame(classification.test.error)
names(error.df.cl) <- c('k','error')
ggplot(data = error.df.cl, aes(x=k,y=error)) + 
  geom_line() + 
  xlab('number of neck neurons') +
  ylab('classification test error')

# Initial index
index <- 1
for (k in seq(20, 400, 20)){
  # Students: need to write up code here
  ae.model <- h2o.deeplearning(    
    x = 2:ncol(train.frame), # select all pixels + extra features
    training_frame = train.frame, # specify the frame (imported file)    
    hidden = c(k), # number of layers and their units
    epochs = 100, # maximum number of epoches  
    activation = 'Tanh', # activation function 
    autoencoder = TRUE # is it an autoencoder? No!
  )
  # Combine middle layer feature with labeled.frame data
  middle.layer.feature <- as.h2o(h2o.deepfeatures(ae.model, train.frame, layer=1))
  combined.train.frame <- h2o.cbind(labeled.frame,middle.layer.feature[1:nrow(labeled.frame),])
  

  NN.model <- h2o.deeplearning(    
    x = 2:ncol(combined.train.frame), # select all pixels + extra features
    y = 1,
    training_frame = combined.train.frame, # specify the frame (imported file)    
    hidden = c(k,k,k), # number of layers and their units
    epochs = 50, # maximum number of epoches  
    activation = 'Tanh', # activation function 
    autoencoder = FALSE, # is it an autoencoder? Yes!
    l2 = 0.1
  )
    # record reconstruction error
  reconstruction.test.error [index,1] <- k
  labeled.predict <- h2o.predict(NN.model, labeled.frame)$predict
  reconstruction.test.error [index,2] <-error.rate(labeled.frame$label, labeled.predict)
  index <- index + 1  
}

# transform extra feture model classification error matrix to data frame 
error.df.st  <- as.data.frame(cbind(classification.test.error,reconstruction.test.error[,2]))


names(error.df.st) <- c('k','error_cl','error_st')

ggplot(data = error.df.st,aes(x=k))+geom_line(aes(y=error_cl ,color="red"))+geom_line(aes(y=error_st,color="blue"))+xlab('number of neck neurons')+ylab('test error')+scale_color_hue(labels = c("3 layers NN", "self taught"))


