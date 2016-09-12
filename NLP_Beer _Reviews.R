#################################################################################################
## OZKAN EMRE OZDEMIR                                                                           #
## HOMEWORK 8 :NLP (Lecture 8)                                                                  #
## 05/30/16                                                                                     #
## Class:  Methods for Data Analysis                                                            #
#################################################################################################
## Clear objects from Memory :
rm(list=ls())
##Clear Console:
cat("\014")

## Get the libraries
library(logging)
library(tm)
library(SnowballC)
library(wordcloud)
library(e1071)
library(caret)
library(RTextTools)
library(topicmodels)
library(slam)
library(RKEA)

library(stringdist)

library(textir)
library(openNLP)
library(openNLPdata)

library(forecast)

# Get the log file name that has a date-time in the name
get_log_filename = function(){
        log_file_name = format(Sys.time(), format="HW8_log_%Y_%m_%d_%H%M%S.log")
        return(log_file_name)
}

# Unit test to check that log file name doesn't exist
test_log_file_name_uniqueness = function(log_file_name){
        all_files = list.files()
        stopifnot(!log_file_name%in%all_files)
}


if (interactive()){
        # Get logger file name
        log_file_name = get_log_filename()
        basicConfig()
        addHandler(writeToFile, file=log_file_name, level='INFO')
        
        # Test for uniqueness
        test_log_file_name_uniqueness(log_file_name)
        
        # Setup working directory
        setwd('~/DataAnalysis/8_NLP')
        
        ##--------BEER REVIEW DATA-----
        # Load scraped beer reviews
        reviews = read.csv("beer_reviews.csv", stringsAsFactors = FALSE)
        reviews$date = as.Date(reviews$date, format = "%Y-%m-%d")
        
        str(reviews)
        range(reviews$date)
        
        # Normalize Data:
        # Change to lower case:
        reviews$review = tolower(reviews$review)
        
        # Remove punctuation
        # Better to take care of the apostrophe first
        reviews$review = sapply(reviews$review, function(x) gsub("'", "", x))
        # Now the rest of the punctuation
        reviews$review = sapply(reviews$review, function(x) gsub("[[:punct:]]", " ", x))
        
        # Remove numbers
        reviews$review = sapply(reviews$review, function(x) gsub("\\d","",x))
        
        # Remove extra white space, so we can split words by spaces
        reviews$review = sapply(reviews$review, function(x) gsub("[ ]+"," ",x))
        
        # Remove non-ascii
        reviews$review = iconv(reviews$review, from="latin1", to="ASCII", sub="")
        
        # remove stopwords
        stopwords()
        my_stops = as.character(sapply(stopwords(), function(x) gsub("'","",x)))
        my_stops = c(my_stops, "beer", "pour", "glass", "bottle", "head", "one")
        
        
        reviews$review = sapply(reviews$review, function(x){
                paste(setdiff(strsplit(x," ")[[1]],my_stops),collapse=" ")
        })# Wait a minute for this to complete
        
        # Remove extra white space again:
        reviews$review = sapply(reviews$review, function(x) gsub("[ ]+"," ",x))
        
        # Stem words:
        reviews$review_stem = sapply(reviews$review, function(x){
                paste(setdiff(wordStem(strsplit(x," ")[[1]]),""),collapse=" ")
        })
        
        # Remove empty/short reviews:
        sum(nchar(reviews$review_stem)<15)
        reviews = reviews[nchar(reviews$review_stem)>15,]
        
        
        ##-----Text Corpus-----
        # We have to tell R that our collection of reviews is really a corpus.
        review_corpus = Corpus(VectorSource(reviews$review_stem))
        
        # Build a Document Term Matrix
        review_document_matrix = TermDocumentMatrix(review_corpus,control=list(bounds = list(global = c(25,Inf))))
        review_term_matrix = DocumentTermMatrix(review_corpus,control=list(bounds = list(global = c(25,Inf))))
        
        # These two matrices are transposes of each other
        dim(review_term_matrix)
        dim(review_document_matrix)
        
        # Too large to write out, so look at a part of it
        inspect(review_term_matrix[1:5,1:5])
        
        # Too large and too sparse, so we remove sparse terms:
        review_term_small = removeSparseTerms(review_term_matrix, 0.995) # Play with the % criteria, start low and work up
        dim(review_term_small)
        
        # Look at frequencies of words across all documents
        word_freq = sort(colSums(as.matrix(review_term_small)))
        
        # Most common:
        tail(word_freq, n=10) #"glass", "bottle", "head", "one" are added to the my_stops
        
        # Least Common:
        head(word_freq, n=10)
        
        # Save Matrix 
        review_corpus_mat = as.matrix(review_term_small)
        dim(review_corpus_mat)
        
        # Convert to Data Frame
        review_frame = as.data.frame(review_corpus_mat)
        head(review_frame)
        # Use only the most common terms
        which_cols_to_use = which(colSums(review_corpus_mat) > 10)
        
        review_frame = review_frame[,which_cols_to_use]
        
        # Convert to factors:
        review_frame = as.data.frame(lapply(review_frame, as.factor))
        
        # Add the rating
        # First, take care of the missing review scores
        reviews$rating[is.na(reviews$rating)] <- mean(reviews$rating, na.rm=T)
        
        # (A1) Normalize "rating" to be between 0 and 1
        reviews$rating = reviews$rating / max(reviews$rating)
        summary(reviews$rating)
        
        review_frame$Rating = reviews$rating
        
        head(review_frame$Rating)
        
        #(A2) Create a logistic model that predicts a normalized review score between zero and one.
        
        # Compute Logistic Linear Model
        reviews_llm = lm(Rating ~ ., data = review_frame)
        sm <- summary(reviews_llm)
        
        #mean standard error
        MSE <- mean(sm$residuals^2)
        
        #mean abs persentage error
        MAPE <- accuracy(reviews_llm)[5]
       
        #answers
        sm # Linear Model Summary
        MSE # 0.005505835
            # Mean Standard Error to estimate the standard deviation of a sampling distribution
            # The standard error of the mean decreases as sample size increases. 
            # The standard error of the mean tells us how the mean varies with different experiments
            # measuring the same quantity.
        MAPE # Mean Abs Persentage Error
            # The Mean Absolute Percentage Error (MAPE), also known as mean absolute percentage deviation (MAPD),
            # measures the accuracy of a method for constructing fitted time series values in statistics.
        # Log Results
        loginfo(paste('# OZKAN EMRE OZDEMIR # HOMEWORK 8 # \n',
                      'The Adjusted R-squared of the linear model was calculated as', summary(reviews_llm)$adj.r.squared, '\n',
                      ', which is very low. However, when we check the Mean Standard Error of the linear model, \n',
                      'MSE =', MSE, '. This value tells us  how the mean varies with different reviwes measuring the same rating. \n',
                      ' It should be noted that the MSE decreases as sample size increase therefore we have a very low MSE value \n',
                      'In addition the Mean Absolute Percentage Error (MAPE) is obtained as' , MAPE,  '\n',
                      '# END #'))
        
}
#############################           End             #############################  