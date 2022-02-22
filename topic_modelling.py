
import gensim
import gensim.corpora as corpora
import numpy as np
from pprint import pprint


########################
#    TOPIC MODELING.   #
########################
def lda_topic_model(tokens, topic_num):
  # Create Dictionary
  id2word = corpora.Dictionary(tokens)
  # Create Corpus
  texts = tokens
  # Term Document Frequency
  corpus = [id2word.doc2bow(text) for text in texts]
  # number of topics
  num_topics = topic_num
  # Build LDA model
  lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
  # Print the Keyword in the 10 topics
  return lda_model


def show_topics(lda_model,word_num_per_topic ):
  for topic in lda_model.print_topics(num_words=word_num_per_topic):
    print("[Topic " + str(topic[0]) +"]: " + topic[1] )

def filter_corpus_by_topic(corpus, tokens, lda_model, topic_index_wanted):
    id2word = corpora.Dictionary(tokens)
    corpus_2 = [id2word.doc2bow(text) for text in tokens]
    topic_probabilities = lda_model[corpus_2]
    topic_predictions = []
    for p in topic_probabilities:
        prediction = p[np.argmax(p, axis=0)[1]][0]
        topic_predictions.append(prediction)
    topic_predictions = np.array(topic_predictions)
    corpus = np.array(corpus)
    return corpus[topic_predictions==topic_index_wanted]


def show_example_sentences_by_topic(corpus, tokens, lda_model, word_num_per_topic,topic_to_check, num_of_examp_to_show):

  print("\n\n######################\n# TOPIC and Examples #\n######################\n")
  print("[Topic " + str(topic_to_check) + "]: " + lda_model.print_topics(num_words=word_num_per_topic)[topic_to_check][1])
  
  filtered_corpus = filter_corpus_by_topic(corpus, tokens, lda_model, topic_to_check)
  for s in filtered_corpus[0:num_of_examp_to_show]:
      print("\n")
      print(s)