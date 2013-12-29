# required packages

require(plyr)
require(rjson)
require(RTextTools)
require(useful)
require(e1071)
require(SparseM)
require(tm)
require(stringr)
require(XML)
require(randomForest)
require(caret)
require(RWeka)

### Set the current working directory and load in the Kaggle Data ###
setwd("/Users/michaelpiccirilli/Desktop/Fall_2013/W4242/Kaggle_Competition")
train.kaggle <- read.table("train.tsv",header=T,sep="\t", stringsAsFactors = FALSE)
test.kaggle <- read.table("test.tsv",header=T,sep="\t",stringsAsFactors=FALSE)


# Strip out the title, body, and URL from the boilerplate
jsonDataTrain <- sapply(train.kaggle$boilerplate, fromJSON)
train.kaggle$title <- sapply(1:nrow(train.kaggle), function(i, jsonDataTrain) unlist(jsonDataTrain[[i]])["title"], jsonDataTrain)
train.kaggle$body <- sapply(1:nrow(train.kaggle), function(i, jsonDataTrain) unlist(jsonDataTrain[[i]])["body"], jsonDataTrain)
train.kaggle$bp_url <- sapply(1:nrow(train.kaggle), function(i, jsonDataTrain) unlist(jsonDataTrain[[i]])["url"], jsonDataTrain)

jsonDataTest <- sapply(test.kaggle$boilerplate, fromJSON)
test.kaggle$title <- sapply(1:nrow(test.kaggle), function(i, jsonDataTest) unlist(jsonDataTest[[i]])["title"], jsonDataTest)
test.kaggle$body <- sapply(1:nrow(test.kaggle), function(i, jsonDataTest) unlist(jsonDataTest[[i]])["body"], jsonDataTest)
test.kaggle$bp_url <- sapply(1:nrow(test.kaggle), function(i, jsonDataTest) unlist(jsonDataTest[[i]])["url"], jsonDataTest)


# Get the classifer, will need this later. 
train.kaggle.classifer <- data.frame(as.factor(train.kaggle$label))
colnames(train.kaggle.classifer)[1] <- "label.classifer"


#### Read in the 2-3gramLength DTM #### 
setwd("/Users/michaelpiccirilli/Desktop/")
nb.predict.body.n2 <- read.csv("submission22-ngram2.csv",header=T,sep=",")
nb.predict.body.n3 <- read.csv("submission22-ngram3.csv",header=T,sep=",")
# code for creating them is directly below




####  2-gramLength DTM on Body text #### 
# create matrices, and NB/prediction
train.body.kaggle.matrix <- create_matrix(train.kaggle$body, language = "english",
                                          removeNumbers=TRUE, removeStopwords = TRUE,
                                          removePunctuation=TRUE, minWordLength=3,
                                          removeSparseTerms=.9950,
                                          toLower=TRUE, ngramLength=2)
train.body.kaggle.matrix <- apply(as.matrix(train.body.kaggle.matrix),2,as.character)
train.body.kaggle.matrix[train.body.kaggle.matrix>1] <- 1
train.body.kaggle.matrix <- as.data.frame(train.body.kaggle.matrix)
train.body.kaggle.matrix$label.classifer <- train.kaggle.classifer$label.classifer

#Test:
test.body.kaggle.matrix <- create_matrix(test.kaggle$body, language = "english",
                                           removeNumbers=TRUE, removeStopwords = TRUE,
                                           removePunctuation=TRUE, minWordLength=3,
                                           removeSparseTerms=.995,
                                           toLower=TRUE, ngramLength=2)
test.body.kaggle.matrix <- apply(as.matrix(test.body.kaggle.matrix),2,as.character)
test.body.kaggle.matrix[test.body.kaggle.matrix>1] <- 1
test.body.kaggle.matrix <- as.data.frame(test.body.kaggle.matrix)


nb.train.body.model <- naiveBayes(label.classifer ~., data=train.body.kaggle.matrix)
nb.test.body.predict <- predict(nb.train.body.model,test.body.kaggle.matrix, type="raw")
nb.test.body.predict <- round(data.frame(nb.test.body.predict))
nb.predict.body.n2 <- round(data.frame(nb.test.body.predict),10)


#### 3-gramLength DTM on Body text ####
# Creates matrices and nB/Preditions
train.body.kaggle.matrix.2 <- create_matrix(train.kaggle$body, language = "english",
                                            removeNumbers=TRUE, removeStopwords = TRUE,
                                            removePunctuation=TRUE, minWordLength=3,
                                            removeSparseTerms=.995,
                                            toLower=TRUE, ngramLength=3)
train.body.kaggle.matrix.2 <- apply(as.matrix(train.body.kaggle.matrix.2),2,as.character)
train.body.kaggle.matrix.2[train.body.kaggle.matrix.2>1] <- 1
train.body.kaggle.matrix.2 <- as.data.frame(train.body.kaggle.matrix.2)
train.body.kaggle.matrix.2$label.classifer <- train.kaggle.classifer$label.classifer

#Test:
test.body.kaggle.matrix.2 <- create_matrix(test.kaggle$body, language = "english",
                                            removeNumbers=TRUE, removeStopwords = TRUE,
                                            removePunctuation=TRUE, minWordLength=3,
                                            removeSparseTerms=.995,
                                            toLower=TRUE, ngramLength=3)
test.body.kaggle.matrix.2 <- apply(as.matrix(test.body.kaggle.matrix.2),2,as.character)
test.body.kaggle.matrix.2[test.body.kaggle.matrix.2>1] <- 1
test.body.kaggle.matrix.2 <- as.data.frame(test.body.kaggle.matrix.2)

nb.train.body.model.2 <- naiveBayes(label.classifer ~., data=train.body.kaggle.matrix.2)
nb.test.body.predict.2 <- predict(nb.train.body.model.2,test.body.kaggle.matrix.2)
nb.test.body.predict.2a <- predict(nb.train.body.model.2,test.body.kaggle.matrix.2,type="raw")
nb.predict.body.n3 <- round(data.frame(nb.test.body.predict.2a),10)

#write these to hd so don't have to re-run
setwd("/Users/michaelpiccirilli/Desktop/")
write.table(nb.predict.body.n2,file="submission22-ngram2.csv",sep=",",row.names=F)
write.table(nb.predict.body.n3,file="submission22-ngram3.csv",sep=",",row.names=F)



#### Bag of words on Body text ####
# create matrices and nB/Prediction
dtm.control.1 <- list( tolower= T, removePunctuation= T, removeNumbers= T,
                       stopwords= c(stopwords("english")), stemming=T, wordLengths= c(3,Inf),
                       weighting= weightTf)

train.body.kaggle.matrix.3 <- DocumentTermMatrix(Corpus(VectorSource(train.kaggle$body)),control = dtm.control.1)
dim(train.body.kaggle.matrix.3)
train.body.kaggle.matrix.3 <- removeSparseTerms(train.body.kaggle.matrix.3,0.995)
train.body.kaggle.matrix.3 <- apply(as.matrix(train.body.kaggle.matrix.3),2,as.character)
train.body.kaggle.matrix.3[train.body.kaggle.matrix.3>1] <- 1
train.body.kaggle.matrix.3 <- as.data.frame(train.body.kaggle.matrix.3)
train.body.kaggle.matrix.3$label.classifer <- train.kaggle.classifer$label.classifer

test.body.kaggle.matrix.3 <- DocumentTermMatrix(Corpus(VectorSource(test.kaggle$body)),control = dtm.control.1)
test.body.kaggle.matrix.3 <- removeSparseTerms(test.body.kaggle.matrix.3,0.9950)
dim(test.body.kaggle.matrix.3)
test.body.kaggle.matrix.3 <- apply(as.matrix(test.body.kaggle.matrix.3),2,as.character)
test.body.kaggle.matrix.3[test.body.kaggle.matrix.3>1] <- 1
test.body.kaggle.matrix.3 <- as.data.frame(test.body.kaggle.matrix.3)

nb.train.body.model.3 <- naiveBayes(label.classifer ~., data=train.body.kaggle.matrix.3)
nb.test.body.predict.3 <- predict(nb.train.body.model.3,test.body.kaggle.matrix.3, type="raw")

nb.predict.body.n1 <- round(data.frame(nb.test.body.predict.3),10)



#### Create the submissions ####

##### Submission 22 ####
submission <- data.frame(test.kaggle$urlid)
colnames(submission)[1] <- "urlid"
sapply(submission,class)
submission$n2 <- nb.predict.body.n2$X1
submission$n3 <- nb.predict.body.n3$X1
submission$na <- as.numeric(submit21$label)
submission$label <- round((((submission$n2 + submission$n2)/2) + submission$na)/2,10) 
# I made a mistake in this submission:   ^-----------------^  As you can see I took
# the 2-gramLength probabilities twice

submit22 <- submission[,c(1,5)]

setwd("/Users/michaelpiccirilli/Desktop/")
write.table(submit22,file="submit22.csv",sep=",",row.names=F)


##### Submission 23 ####
submit23 <- data.frame(test.kaggle$urlid)
colnames(submit23)[1] <- "urlid"
submit23$n2 <- nb.predict.body.n2$X1
submit23$n3 <- nb.predict.body.n3$X1
submit23$n1 <- nb.predict.body.n1$X1
submit23$label <- round((((submit23$n2 + submit23$n3)/2) + submit23$n1)/2,10)
submit23 <- submit23[,c(1,6)]

setwd("/Users/michaelpiccirilli/Desktop/")
write.table(submit23,file="submit23.csv",sep=",",row.names=F)

