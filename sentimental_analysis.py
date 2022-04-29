"""
@author: Team stonks
"""
import requests
import flair
import regex as re
import pandas as pd


def get_data(tweet):                                                           #get data from twitter
    data = {       
        'id': tweet['id_str'],                                                 #extracting id of tweet
        'created_at': tweet['created_at'],                                     #timestamp of tweet
        'text': tweet['full_text']                                             #content of tweet
    }
    return data                                                              



def Senti_analyze(company):                                                    #sentimental Anlysis
    params = {                                                                 #parameter
    'q': company,                                                              #company name or acronym
    'tweet_mode': 'extended',                                                  #to extract full tweet 
    'lang': 'en',                                                              #language of tweet
    'count':'10000'                                                            #number of tweets to extract
}


    BEARER_TOKEN='AAAAAAAAAAAAAAAAAAAAAH3BbAEAAAAANhrInokh03CpxLwZNoSP1qfpQSc%3Dn4eeAxqx51TqnbPu2w60mJ0B51tswZH9l6kX7Nn4phyO8DoLrp'


    response=requests.get(
        'https://api.twitter.com/1.1/search/tweets.json',                      #initiating request for tweets
        params=params,
        headers={'authorization': 'Bearer '+BEARER_TOKEN}                      #For authorization
        )



    df = pd.DataFrame()                                                        #insert tweets to dataframe
    for tweet in response.json()['statuses']:                                  #Json key
        row = get_data(tweet)
        df = df.append(row, ignore_index=True)                                 #append in dataframe ignoring indices 
    
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')         #initializing flair model     
    tweets=df['text']                                                          #tweets in dataframe for screening

    probs = []                                                                 #probablity
    sentiments = []                                                            #sentiment list


    for tweet in tweets.to_list():
        sentence = flair.data.Sentence(tweet)                                  # make prediction
        sentiment_model.predict(sentence)                                      # extract sentiment prediction
        probs.append(sentence.labels[0].score)                                 # numerical score 0-1
        sentiments.append(sentence.labels[0].value)                            # 'POSITIVE' or 'NEGATIVE'

                                                                               

    tweets['probability'] = probs                                              # add probability and sentiment predictions to tweets dataframe
    tweets['sentiment'] = sentiments


    senti=pd.DataFrame()                                                       #dataframe
    senti['date']=df['created_at']                                             #inserting creation date 

    senti['sentiment']=sentiments[0:]

    senti['Probability']=probs[0:]
    
    return senti





