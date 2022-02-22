import pandas as pd
from preprocessing import preprocess
from wordclouds import wordcloud, find_by_word
from sentiment_analysis import calculate_sentiment, find_by_sentiment
import nltk
import os
import tempfile
from topic_modelling import lda_topic_model, show_topics, show_example_sentences_by_topic

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
nltk.download('punkt')
nltk.download('stopwords')
pd.set_option('display.max_columns', None)

#####################################################
#               READING THE DATA                    #
#####################################################
# In this tutorial we will mostly deal with comma separated files (CSV) (similar to the structure of Excel files). Each line of the file is a data record. Each record consists of one or more fields, separated by commas. Check here for more information https://en.wikipedia.org/wiki/Comma-separated_values 

# reads the file named "students_eng.csv". 
# If you want to read a different file you need to (1) upload it in replit and (2) change "students_eng.csv" to the name of the newly uploaded file. Here we use the Pandas library ("pd") to read our file and in return we get a Pandas Dataframe. For faster processing and experimentation you can also select different subsets of the file's content through the nrows parameter -> number of lines to read.
students_data = pd.read_csv("data/students_eng.csv") 


# With the next line you can print the data you just read and see how a Pandas Dataframe looks like (seems quite similar to Excel)

print(students_data.head(3))

# As you can see the data is separated in columns. Let's see how we can get the data from a specific column. The following line allows us to get only the data inside the column named "students_needs". Other options are: study_programme, degree_programme, planned_grad_year, today_feeling, physical_health, student_needs, students_sugg_to_improve_wellbeing

students_data = students_data['student_needs']

#################################################
#         TEXT PREPROCESSING                    #
#################################################

# Here we will pre-process our entire text collection. 
# First, we need to merge all the different lines of the "comments" into one big corpus, so that we can later analyze it.

corpus = students_data.to_list()

print(corpus[0:5])

# Then we need to "preprocess" our text. To do so we use the following line of code (more details on what happens under the hood could be found in the  "preprocessing.py" file - feel free to take a look at it).

# The following code: makes all words lowercase, create word tokens, removes stopwords, punctuations, and digits, and reduces inflected words to their word stem (stemming).Feel free to experiment by turning any of the following values from True to False. In addition, you can add extra words which you do not want to include in your analysis by adding them within the extra_stopwords brackets e.g. extra_stopwords=["people"] would remove the word people from everywhere in the document. Hint: don't forget to use the quotes!


# tokens = [preprocess(sentence, lower=True, rem_punc=True, word_tokenization=True, rem_numb=True, rem_stopwords=True, stem=True, extra_stopwords = []) for sentence in students_data.to_list()]
# print(tokens)


#############################################
#             WORD FREQUENCIES              #
#############################################

# Word frequencies calculation is the most basic tool in text processing yet it gives a comprehensive picture of the content in your text collection. One the most ways to visualize word frequencies is WordCloud (which you've already seen if you opened Voyant) 

# This function needs two things from you:
# 1. tokens -- the result of our preprocessing step
# 2. the name of the picture it will generate and save to your directory
# 3. Number of words to show


# wordcloud(words = tokens, name_of_output = 'wordcloud', num = 10)

# Text processing often requires working with examples, because words are often contextual and it is difficult to understand what is happening in your text collection. For this purpose, you can find documents by pieces of texts. 

# This function needs two things from you:
# 1. tokens -- the result of our preprocessing step (it will look for examples in this collection)
# 2. a word or a phrase the text should include

# test = find_by_word(tokens, 'studi')
#print(test)

#############################################
#            Sentiment analysis             #
#############################################

# The aim of sentiment analysis is to calculate how emotional your texts are and what is the valence of these texts. In our example we use VADER (Valence Aware Dictionary and sEntiment Reasoner) but you can find other various sentiment analysis tools in the internet. 

# VADER calculated how positive, neutral, and negative a text is. It also calculates a compound score which considers all three metrics to give you a precise measurement of the sentiment.

# This function requires only the preprocessed collection of texts

# sent_result = calculate_sentiment(tokens)
# print(sent_result)

# Now, when the sentiment scores are calculated, you can find the most interesting texts by looking at the documents with highest scores (in this example, we look at the 5 most positive documents).

# This function requires three things:
# 1. The result of sentiment calculation
# 2. What score you're interested in
# 3. Number of examples you want to get

# res = find_by_sentiment(df_with_scores = sent_result, score_type = 'pos', num_of_examples = 5)
# print(res)

#############################################
#         TOPIC MODELING                    #
#############################################
# num_of_topics = 4
# word_num_per_topic = 5

# lda_model = lda_topic_model(tokens, topic_num=num_of_topics)
# show_topics(lda_model, word_num_per_topic )


# Check examples assigned to a particular topic ####
# num_of_examples = 5

# show_example_sentences_by_topic(corpus, tokens, lda_model, word_num_per_topic,topic_to_check=1, num_of_examp_to_show = num_of_examples)