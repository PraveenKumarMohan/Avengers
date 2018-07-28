library(jpeg)

#Path of the data from where the images should be fetched
Data.dir <- 'C:/Study Materials/Machine Learning/Avengers/All'

#Get all the file names in a vector
pic1 <- as.vector(list.files(path = Data.dir, full.names = TRUE, include.dirs = FALSE))

mydata <- c()

#Read al the JPEG files
for (i in 1:length(pic1)) {mydata[[i]] <- readJPEG(pic1[[i]])}

#Resize the images to 28*28 and convert them to a grayscale image
for (i in 1:length(mydata)) {mydata[[i]] <- resize(Image(data = mydata[[i]], dim = dim(mydata[[i]]), colormode = "Grayscale"), 28, 28)}

#Getting the features of the images in a vector
for (i in 1:length(mydata)) {mydata[[i]] <- as.vector(mydata[[i]])}

#Converting the vector to a data frame for further processing
df <- as.data.frame(do.call(rbind, mydata))

#Read the labels of the images from a csv
lab <- read.csv(file = "C:/Study Materials/Machine Learning/Avengers/labels.csv", header = T)

#Merge the label vector to the data frame we have created earlier
df <- data.frame(cbind(lab$Label, df))
table(df$lab.Label)


#Changing the labels of the dependent variable
df$lab.Label <- as.numeric(factor(df$lab.Label, levels = c("Black Widow", "Captain America", "Hulk", "Iron Man"), labels = c(0,1,2,3)))
names(df)[1] <- c("Label")
str(df)

#Check if any feature could be removed
columnsKeep <- names(which(colSums(df[,-1]) > 0))
df <- df[c("Label", columnsKeep)]

library(caret)
library(e1071)
library(ggfortify)

set.seed(17125201)
idx <- createDataPartition(df$Label, p=0.75, list = FALSE)

#PCA on the features
pca <- prcomp(df[idx,-1], scale. = F, center = F)
autoplot(pca, data = df[idx,], colour='Label')
screeplot(pca, type = "lines", npcs = 50)

var.pca <- pca$sdev ^ 2
x.var.pca <- var.pca/sum(var.pca)
cum.var.pca <- cumsum(x.var.pca)

#Graph to determine the number of pcs required
plot(cum.var.pca[1:30], xlab = "No. of PCs", 
     ylab = "Cumulative Proportion of variance explained", ylim = c(0,1), type = 'b')

#PCA rotation on the data
pcs <- 20
indata <- as.data.frame(as.matrix(df[,-1]) %*% pca$rotation[,1:pcs])
indata <- data.frame(cbind(df[,1], indata))
names(indata)[1] <- c("Label")
str(indata)

#Splitting the data into train and test
train <- indata[idx,]
test <- indata[-idx,]

#Hyperparameter Optimization using Deep Learning
library(h2o)
hidden_opt <- list(c(100, 100, 100, 100), c(200, 200, 200, 200), c(300, 300, 300, 300)) 
l1_opt <- c(1e-5,1e-7)
activations <- c("Tanh", "TanhWithDropout", "Rectifier", "RectifierWithDropout", "Maxout", "MaxoutWithDropout")
hyper_params <- list(hidden = hidden_opt, l1 = l1_opt, activation=activations)

h2o.init(ip = "localhost", port = 54321)
train$Label <- factor(train$Label)
h2otrain <- train
h2otrain <- as.h2o(h2otrain)
#test$Label <- as.integer(test$Label)
h2otest <- test
h2otest <- as.h2o(h2otest)

model_grid <- h2o.grid("deeplearning",
                       hyper_params = hyper_params,
                       x = c(2:length(h2otrain)),  # column numbers for predictors
                       y = 1,   # column number for label
                       training_frame = h2otrain,
                       validation_frame = h2otest)

dlPerf <- c()

for (model_id in model_grid@model_ids){
  model <- h2o.getModel(model_id)
  pred <- h2o.predict(model, h2otest)
  pred <- as.data.frame(pred)
  dlPerformance <- 1 - mean(pred$predict != test$Label)
  dlPerf <- rbind(dlPerf, dlPerformance)
}
#The best accuracy
(bestDL <- max(dlPerf))
