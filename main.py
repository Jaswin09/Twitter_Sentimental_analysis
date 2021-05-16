import csv

import credentials
import twitter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import re

twitter_api=twitter.Api(consumer_key=credentials.API_KEY,
                        consumer_secret=credentials.API_SECRET,
                        access_token_key=credentials.ACCESS_TOKEN,
                        access_token_secret=credentials.ACCESS_SECRET)

#print(twitter_api.VerifyCredentials())

def build_testset(keyword):
    try:
        tweets=twitter_api.GetSearch(keyword,count=100)
        #print(tweets)
        print("Fetched "+str(len(tweets))+" tweets for the keyword "+keyword)
        return [{"text": i.text, "label": None} for i in tweets]
    except:
        print("Sorry,we couldnt find anything for you")
        return None

searchword = input("Enter the search keyword : ")
test_data_set=build_testset(searchword)
#print(test_data_set)

training_data_set = []
with open("tweetDataFile.csv", 'rt',encoding="utf8") as csvfile:
    lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
    for row in lineReader:
        training_data_set.append({"tweet_id": row[0], "text": row[1], "label": row[2], "topic": row[3]})

class PreProcesstweets:
    def __init__(self):
        self.stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def process_tweets(self, list_of_tweets):
        processed_tweets = []
        for tweet in list_of_tweets:
            if tweet["label"] is not None:
                if tweet["label"] == "positive" or tweet["label"] == "negative":
                    processed_tweets.append((self._process_tweet(tweet["text"]), tweet["label"]))
            else:
                processed_tweets.append((self._process_tweet(tweet["text"]), None))

        return processed_tweets

    def _process_tweet(self, tweet):
        tweet=tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
        tweet = word_tokenize(tweet)

        words=[]
        for i in tweet:
            if i not in self.stopwords:
                words.append(i)
        return words

tweet_processor = PreProcesstweets()
training_set = tweet_processor.process_tweets(training_data_set)
testing_set = tweet_processor.process_tweets(test_data_set)

def build_vocabulary(data):
    all_words=[]
    for (words,sentiment) in data:
        all_words.extend((words))
    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    return word_features

training_data_features = build_vocabulary(training_set)

def extract_features(tweet):
    tweet_words = set(tweet)
    features={}
    for i in training_data_features:
        is_feature_in_words = i in tweet_words
        features[i]=is_feature_in_words
    return features

training_features = nltk.classify.apply_features(extract_features,training_set)

Nbayesclassifier = nltk.NaiveBayesClassifier.train(training_features)

classified_result_labels = []
for i in testing_set:
    classified_result_labels.append(Nbayesclassifier.classify(extract_features(i[0])))

#print(classified_result_labels.count('positive'))
#print(classified_result_labels.count('negative'))

if classified_result_labels.count('positive') > classified_result_labels.count('negative'):
    print("Overall Positive sentiment")
    print("Positive sentiment percentage = " + str(((classified_result_labels.count('positive'))/len(classified_result_labels))*100))
else:
    print("Overall Negative sentiment")
    print("Negative sentiment percentage = " + str(((classified_result_labels.count('negative'))/len(classified_result_labels))*100))