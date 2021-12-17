# Import the required packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import streamlit as st
import pickle
from pickle import load
from PIL import Image
import seaborn as sns
import statsmodels.api as sm
import lime.lime_tabular
from sklearn.model_selection import train_test_split
import string
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
# Set Recursion Limit
import sys
sys.setrecursionlimit(40000)
import re  
import nltk  
import regex as re
from nltk.corpus import stopwords  
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
import streamlit.components.v1 as components
import tweepy
from collections import Counter
from wordcloud import WordCloud
import datetime
import plotly.express as px
import time
import pydeck as pdk

st.sidebar.title('Dashboard Control')
control = st.sidebar.radio('Navigation Bar', ('Home', 'Live Tweet Feed', 'Time Series Analysis', 'XAI'))

if control == 'Home':
    ### Sentiment Code goes here
    
    st.markdown('<h1 style="color:#8D3DAF;text-align:center;font-family: Garamond, serif;"><b>RAKSHAK</b></h1>',unsafe_allow_html=True)
    st.markdown('<h2 style="color:#E07C24;text-align:center;font-family: Georgia, serif;"><b>Time Series Sentiment Analysis Of Natural Hazard Relief Operations Through Social Media Data</b></h2>',unsafe_allow_html=True)
    
    #st.markdown("The dashboard will help the government and humanitarian aid agencies to plan and coordinate the natural disaster relief efforts, resulting in more people being saved and more effective distribution of emergency supplies during a natural hazard")

    st.header("Natural Hazard Data Collected Sample")
    # Dataset
    # Load the Dataset
    tweets1 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/nepal_mix_1.csv")[['text','type']]
    tweets2 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/italy_mix_1.csv")[['text','type']]
    tweets3 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/Covid-19.csv")[['text','type']]
    names = [tweets1,tweets2,tweets3]
    # Concatenate the datasets
    tweets = pd.concat(names,ignore_index = True)
    # Reshuffle the dataset
    tweets = tweets.sample(frac = 1)
    # Reindex the dataset
    tweets['index'] = list(range(0,tweets.shape[0],1))
    tweets.set_index('index', inplace=True)
    tweets['type'] = tweets['type'].map({0: 'Need', 1: 'Availability', 2: 'Other'})
    # Change column names for consistency
    tweets.columns = ['text', 'type']
    # Dataset Description
    h = st.sidebar.slider('Select the number of tweets using the slider', 1, 100, 10)
    data_tweets = tweets.sample(h)
    data_tweets['index'] = list(range(0, h, 1))
    data_tweets.set_index('index', inplace=True)
    st.table(data_tweets)
    
    
    # Checking for class balancing and get unique labels:
    st.header("Count Of Tweets In Each Class")
    chart_visual_class_balancing = st.sidebar.checkbox('Class Labels', True)
    if chart_visual_class_balancing==True:
        fig = plt.figure(figsize=(8, 4))
        #sns.countplot(y=tweets.loc[:, 'type'],data=tweets).set_title("Count of tweets in each class")
        fig = px.histogram(tweets, x="type",color="type",title="Count of tweets in each class")
        st.plotly_chart(fig)
        
    # Wordclouds
    # Selection of Input & Output Variables
    X = tweets.loc[:, 'text']
    Y = tweets.loc[:, 'type']
    X = list(X)
    def preprocess_dataset(d):    
        # Define count variables
        cnt=0
        punctuation_count = 0
        digit_count = 0
            
        # Convert the corpus to lowercase
        lower_corpus = []
        for i in range(len(d)):
            lower_corpus.append(" ".join([word.lower() for word in d[i].split()]))
                        
        # Remove any special symbol or punctuation
        without_punctuation_corpus = []
        for i in range(len(lower_corpus)):
            p = []
            for ch in lower_corpus[i]:
                if ch not in string.punctuation:
                    p.append(ch)
                else:
                    p.append(" ")
                    # Count of punctuation marks removed
                    punctuation_count += 1
            x = ''.join(p)
            if len(x) > 0:  
                without_punctuation_corpus.append(x)
          
        # Remove urls with http, https or www and Retweets RT
        without_url_corpus = []
        for i in range(len(without_punctuation_corpus)):
            text = without_punctuation_corpus[i]
            text = re.sub(r"http\S*||www\S*", "", text)
            text = re.sub(r"RT ", "", text)
            without_url_corpus.append(text)
            
        # Remove special characters and numbers from the corpus
        without_digit_corpus = []
        for i in range(len(without_url_corpus)):
            p = []
            for word in without_url_corpus[i].split():
                if word.isalpha():
                    p.append(word)
                else:
                    # Count of punctuation marks removed
                    digit_count += 1
            x = ' '.join(p)
            without_digit_corpus.append(x)
            
                
        # Tokenize the corpus
        # word_tokenize(s): Tokenize a string to split off punctuation other than periods
        # With the help of nltk.tokenize.word_tokenize() method, we are able to extract the tokens
        # from string of characters by using tokenize.word_tokenize() method. 
        # Tokenization was done to support efficient removal of stopwords
        total_count = 0
        tokenized_corpus = []
        for i in without_digit_corpus:
            tokenized_tweet = nltk.word_tokenize(i)
            tokenized_corpus.append(tokenized_tweet)
            # Count the length of tokenized corpus
            total_count += len(list(tokenized_tweet))
                
        # Remove Stopwords
        stopw = stopwords.words('english')
        count = 0
        tokenized_corpus_no_stopwords = []  
        for i,c in enumerate(tokenized_corpus): 
            tokenized_corpus_no_stopwords.append([])
            for word in c: 
                if word not in stopw:  
                    tokenized_corpus_no_stopwords[i].append(word) 
                else:
                    count += 1
    
        # lemmatization and removing words that are too large and small
        lemmatized_corpus = []
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        ct = 0
        cnt_final=0
        dictt = {}
        for i in range(0,len(tokenized_corpus_no_stopwords)):
            lemmatized_corpus.append([])
            for w in tokenized_corpus_no_stopwords[i]:
                # lematizing only those words whose length >= 2 and <=10
                # Considering words with length greater than or equal to 2 and less than or equal to 10
                if(len(w)>2 and len(w)<=10):
                    lemmatized_corpus[i].append(lemmatizer.lemmatize(w))
                    cnt_final+=1
                # Count of final corpus
                # This is the length of total corpus that went through the process of lematization
                ct+=1
      
        ############## Removing words of large and small length
        # Doing a survey to find out the length of words so we can remove the too small and too large words from the Corpus
        # plt.bar(*zip(*dictt.items()))
        # plt.show()
    
        # Punctuation Preprocessing
        preprocessed_corpus = []
        for i,c in enumerate(lemmatized_corpus):
            preprocessed_corpus.append([])
            for word in c:
                x = ''.join([ch for ch in word if ch not in string.punctuation])
                if len(x) > 0:
                    preprocessed_corpus[i].append(x)
            
       
        # Clear unwanted data variables to save RAM due to memory limitations
        del lower_corpus
        del without_punctuation_corpus
        del without_digit_corpus
        del tokenized_corpus
        del tokenized_corpus_no_stopwords
        del lemmatized_corpus
        return preprocessed_corpus
    
    # Preprocess the Input Variables
    preprocessed_corpus = preprocess_dataset(X)
    data_corpus = []
    for i in preprocessed_corpus:
        data_corpus.append(" ".join([w for w in i]))

    # Creating a word cloud
    st.header("Wordclouds For Dataset")
    fig, axes = plt.subplots(1, 2)
    # Worcloud for processed dataset
    words1 = ' '.join([tweet for tweet in X])
    words2 = ' '.join([tweet for tweet in data_corpus])
    wordCloud1 = WordCloud(background_color ='black').generate(words1)
    wordCloud2 = WordCloud(background_color ='black').generate(words2)    
    # Display the generated image:
    axes[0].title.set_text("Raw Dataset")
    axes[0].imshow(wordCloud1)
    axes[0].axis("off")
    axes[1].title.set_text("Processed Dataset")
    axes[1].imshow(wordCloud2)
    axes[1].axis("off")
    st.pyplot(fig)
    
    # Create most used hashtags
    st.header("Top Hashtag Used in the Datasets")
    fig, axes = plt.subplots(1, 3)
    tweets1 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/nepal_mix_1.csv")[['text','type']]
    tweets2 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/italy_mix_1.csv")[['text','type']]
    tweets3 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/Covid-19.csv")[['text','type']]
    X1 = list(tweets1.loc[:, 'text'])
    X2 = list(tweets2.loc[:, 'text'])
    X3 = list(tweets3.loc[:, 'text'])
    dc1 = []
    pd1 = preprocess_dataset(X1)
    for i in pd1:
        dc1 += i
    c1 = Counter(dc1)
    mfw1 = c1.most_common(10)
    df1 = pd.DataFrame(mfw1)
    df1.columns = ['Word', 'Count']
    axes[0] = px.line(df1, x='Word', y='Count',title='Nepal Earthquake 2015',labels={'Word':'Hashtag', 'Count':'Number of Hashtag tweeted'})
    st.plotly_chart(axes[0])

                      
    dc2 = []
    pd2 = preprocess_dataset(X2)
    for i in pd2:
        dc2 += i
    c2 = Counter(dc2)
    mfw2 = c2.most_common(10)
    df2 = pd.DataFrame(mfw2)
    df2.columns = ['Word', 'Count']
    axes[1] = px.line(df2, x='Word', y='Count',title='Italy Earthquake 2016', labels={'Word':'Hashtag', 'Count':'Number of Hashtag tweeted'})
    st.plotly_chart(axes[1])
   
                  
    dc3 = []
    pd3 = preprocess_dataset(X3)
    for i in pd3:
        dc3 += i
    c3 = Counter(dc3)
    mfw3 = c3.most_common(10)
    df3 = pd.DataFrame(mfw3)
    df3.columns = ['Word', 'Count']
    axes[2] = px.line(df3, x='Word', y='Count',title='COVID-19',labels={'Word':'Hashtag', 'Count':'Number of Hashtag tweeted'})
    st.plotly_chart(axes[2])

    #df3.set_index('Word', inplace=True)
    #axes[2].plot(df3['Count'], marker='o', linewidth=0.5,ls='solid', c='blue')
    #axes[2].tick_params(axis ='x', rotation =-90)
    #axes[2].set_xlabel('Hashtag')
    #axes[2].set_ylabel('Number of Hashtag tweeted')
    #axes[2].title.set_text("COVID-19")
              
    st.header("Select Start & End Date to display sentiments")
    s_date = st.date_input("Start Date", min_value=datetime.datetime(2021, 4, 1), max_value=datetime.datetime(2021, 4, 30), value=datetime.datetime(2021, 4, 1))
    e_date = st.date_input("End Date", min_value=datetime.datetime(2021, 4, 1), max_value=datetime.datetime(2021, 4, 30), value=datetime.datetime(2021, 4, 30))
    data = pd.read_csv('sentiment_april.csv')[['Need','Availability']]
    data_T = data.T
    date1 = int(str(s_date)[8:])-1
    date2 = int(str(e_date)[8:])  
    data_T["sum"] = data_T[list(range(date1,date2,1))].sum(axis=1)
    l_name = ['Need', 'Availability']
    l_value = data_T['sum']
    pie_dict = {'name': l_name, 'value': l_value}
    pie_df = pd.DataFrame(pie_dict)
    fig_pie = px.pie(pie_df, values='value', names='name', title='Sentiments of tweet collected between '+str(s_date)+' and '+str(e_date))
    st.plotly_chart(fig_pie)
    
    
    # Show locations for tweets
    st.header("Map for Location of Each User")
    df = pd.read_csv('lat-long.csv')
    df.columns = ['lat', 'lon', 'country']
    st.map(df)
 
    
   

elif control == 'Live Tweet Feed':
    ### Libe Tweet feed goes here
    st.markdown('<h1 style="color:Black;text-align:center;"><b>Live Tweet Feed</b></h1>',unsafe_allow_html=True)
        
    st.header("Live Tweet Feed Sample")
    hashtag = str(st.text_input("Enter the keyword or hashtag for live twitter fee", "#coronavirus"))
    fetch_tweets = st.button("Fetch Tweets")
    
    ####input your credentials here
    consumer_key = "IE5dmFVlYdg5aNrsNnZiXZVPa"
    consumer_secret = "29t2ERW1eZELcJcRGrwYQ1ANvSHocf3GFeknkD38wfQyn0dJaz"
    access_token = "1116791204954497024-FVE38zJow8Y3Ut54tIJO30GvLbk1zA"
    access_token_secret = "AOk5BmOFky9Y1Y37IqKjhp3mec9dHaqXV4Mt38mLPtQI5"
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    
    if fetch_tweets:
        # Current Time
        current_time = time.time()
        diff = 0
        real_time = 0
        live_tweet_text = []
        live_tweet_date = []
        live_tweet_id = []
        lt_user_name = []
        lt_user_location = []
        lt_user_screenname=[]
        lt_followers = []
        lt_following = []
        
        while(diff < 10):
            for tweet in tweepy.Cursor(api.search_tweets,q=hashtag,count=10,lang="en",since="2021-12-11").items():
                real_time = time.time()
                diff = real_time - current_time
                if diff >10:
                    break
                if (not tweet.retweeted) and ('RT @' not in tweet.text):
                    #print(tweet,"\n\n\n\n\n")
                    live_tweet_text.append(tweet.text)
                    live_tweet_date.append(tweet.created_at)
                    live_tweet_id.append(tweet.id)
                    lt_user_name.append(tweet.user.name)
                    lt_user_location.append(tweet.user.location)
                    lt_user_screenname.append(tweet.user.screen_name)
                    lt_followers.append(str(tweet.user.followers_count))
                    lt_following.append(str(tweet.user.friends_count))
    
        live_tweet_feed_dict = {'Tweet ID':live_tweet_id, 'Tweet': live_tweet_text, 'Date & Time': live_tweet_date, 'Username': lt_user_screenname, 'User Full Name': lt_user_name, 'Location': lt_user_location, 'Follower Count': lt_followers, 'Following Count': lt_following}      
        live_tweet_feed = pd.DataFrame(live_tweet_feed_dict)
        st.dataframe(live_tweet_feed)
            
    



elif control == 'Time Series Analysis':
    ### Streamlit code starts here    
    st.markdown('<h1 style="color:Black;text-align:center;"><b>Time Series Analysis of Disaster Tweets</b></h1>',unsafe_allow_html=True)
    
    ### Time Series Code goes here
    
    # Dataset
    # Load the Dataset
    tweets1 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/nepal_mix_1.csv")[['text','type']]
    tweets2 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/italy_mix_1.csv")[['text','type']]
    tweets3 = pd.read_csv("https://raw.githubusercontent.com/anidevhere/Temp/main/Covid-19.csv")[['text','type']]
    names = [tweets1,tweets2,tweets3]
    
    # Concatenate the datasets
    tweets = pd.concat(names,ignore_index = True)
    
    # Reshuffle the dataset
    tweets = tweets.sample(frac = 1)
    
    # Reindex the dataset
    tweets['index'] = list(range(0,tweets.shape[0],1))
    tweets.set_index('index', inplace=True)
    tweets['type'] = tweets['type'].map({0: 'Need', 1: 'Availability', 2: 'Other'})
    # Change column names for consistency
    tweets.columns = ['text', 'type']
          
    tweets['type'] = tweets['type'].map({'Need':0, 'Availability':1,'Other':2})
    
    # Get all the labels used in the labelling column
    label = tweets.type.unique()
    print("Labels:", label)
    
    # Remove label 2 from the list because not required for time series analysis
    label = np.delete(label,np.where(label == 2))
    print("Labels:", label)
    
    # Add names to the numerical labels
    label_name = []
    for i in label:
        if i == 0:
            label_name.append("Need")
        elif i == 1:
            label_name.append("Availability")
            
    # Choose interval
    interval = 30
    start_date = "2021-04-01"
    
    # Create Timestamps with intervals
    ds = pd.date_range(start=start_date, periods=interval)
    dates = []
    for i in ds:
        dates.append(i.strftime('%m-%d-%Y'))
    del ds
    
    # Divide the Dataset into intervals
    
    # Divide the dataset into the given number of intervals
    num_of_tweets_per_interval = math.floor(tweets.shape[0]/interval)
    
    # Create Time Series with intervals
    data = []
    count_of_data = []
    for i in label:
        count_of_data.append([])
    
    for i in range(1,interval+1,1):
        # Draw a sample from the tweets
        tw = tweets.sample(n=num_of_tweets_per_interval, random_state=10, replace=False)
        # Append the statistics of the drawn sample to the list
        stat = dict()
        for j in range(0,len(label)):
            stat[label[j]] = list(tw['type']).count(label[j])
            count_of_data[j].append(list(tw['type']).count(label[j]))
        data.append(stat)
        # Remove the already drawn tweets from the dataset
        tweets.drop(labels=list(tw.index.values),inplace=True)
    
    
    # Real Time Series starts here
    # Load Dataset
    df = pd.DataFrame(count_of_data).T
    # Set Index
    df['Date'] = pd.to_datetime(dates)
    df.set_index('Date', inplace=True)
    df.columns = ['Need', 'Availability']
    
    
    st.title("Twitter Data Description")
    chart_visual_tweets = st.selectbox('Select Chart/Plot type', 
                                        ('Stacked Bar Chart', 'Side-by-Side Bar Chart', 'Line Chart'))
    
    # Plot 1
    if chart_visual_tweets=='Side-by-Side Bar Chart':
        # set width of bars
        barWidth = 0.25
        # Set position of bar on X axis
        r = [np.arange(interval)]
        for i in range(1, len(label)):
            r1 = [x + barWidth for x in r[-1]]
            r.append(r1)
        # Plotting a line plot after changing it's width and height
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(8)
        # Make the plot
        for i,lab in enumerate(label):
            plt.bar(r[i], count_of_data[i], width=barWidth, edgecolor='white', label=label_name[i])
        # Add xticks on the middle of the group bars
        plt.xlabel('Time Series', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(count_of_data[0]))], list(dates))
        plt.tick_params(axis ='x', rotation =90)
        # Create legend & Show graphic
        plt.legend()
        plt.show()    
        st.pyplot(f)
    
    # Plot 2
    if chart_visual_tweets=='Stacked Bar Chart':
        # Plotting a line plot after changing it's width and height
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(8)
    
        b = np.zeros(interval)
        for i,lab in enumerate(label):
            plt.bar(dates, count_of_data[i],bottom=b, edgecolor='white', label=label_name[i])
            b += np.array(count_of_data[i])
        
        plt.xlabel('Time Series', fontweight='bold')
        plt.tick_params(axis ='x', rotation =90)
        
        # Create legend & Show graphic
        plt.legend()
        plt.show()
        st.pyplot(f)
    
    
    # Plot 3
    if chart_visual_tweets=='Line Chart':
        # Plotting a line plot after changing it's width and height
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(8)
        ls = ['dashed', 'solid']
        for i,lab in enumerate(label):
            plt.plot(count_of_data[i], label=label_name[i], linestyle=ls[i], marker='o')
        plt.xlabel('Time Series', fontweight='bold')
        plt.tick_params(axis ='x', rotation =90)
        # Create legend & Show graphic
        plt.legend()
        plt.show()
        st.pyplot(f)
    
    
    ################################### Time Series Analysis starts here
    st.title("Time Series Analysis of Tweets")
    chart_visual_time_series = st.radio('Select Need/Availability Label for Time series distribution',('Need', 'Availability')) 
    options = st.multiselect(
     'Select options for Data Resampling',
     ['D', '3D', 'W', '2W'],
     ['D', '3D'])
    # y represemts the Need Label
    # z represents the Availability Label
    y = df['Need']
    z = df['Availability']
    
    if chart_visual_time_series=='Need':
        fig, ax = plt.subplots(figsize=(20, 6))
        if 'D' in options:
            ax.plot(y, marker='o', linewidth=0.5, label='Daily',ls='solid', c='red')
        if '3D' in options:
            ax.plot(y.resample('3D').mean(),marker='o', markersize=8, linestyle='dashed', label='Half-Weekly Mean Resample')
        if 'W' in options:
            ax.plot(y.resample('W').mean(),marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
        if '2W' in options:
            ax.plot(y.resample('2W').mean(),marker='o', markersize=8, linestyle='dotted', label='Bi-weekly Mean Resample')

        ax.set_ylabel('Frequency')
        ax.set_xlabel('Date')
        ax.legend()
        st.pyplot(fig)
    
    if chart_visual_time_series=="Availability":
        fig, ax = plt.subplots(figsize=(20, 6))
        if 'D' in options:
            ax.plot(y, marker='o', linewidth=0.5, label='Daily',ls='solid', c='red')
        if '3D' in options:
            ax.plot(y.resample('3D').mean(),marker='o', markersize=8, linestyle='dashed', label='Half-Weekly Mean Resample')
        if 'W' in options:
            ax.plot(y.resample('W').mean(),marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
        if '2W' in options:
            ax.plot(y.resample('2W').mean(),marker='o', markersize=8, linestyle='dotted', label='Bi-weekly Mean Resample')

        ax.set_ylabel('Frequency')
        ax.set_xlabel('Date')
        ax.legend()
        st.pyplot(fig)
    
    
    ################################### Seasonal Decomposition starts here
    # The next step is to decompose the data to view more of the complexity behind the linear visualization. 
    # A useful Python function called seasonal_decompose within the 'statsmodels' package can help us to decompose the data 
    # into four different components:
    # Observed
    # Trended
    # Seasonal
    # Residual
    st.title("Decompose the Data")
    chart_visual_seasonal_decomposition = st.radio('Select Need/Availability Label for Seasonal decomposition', 
                                        ('Need of resources', 'Availability of resources'))
    
    
    def seasonal_decompose (x):
        decomposition_x = sm.tsa.seasonal_decompose(x, model='additive',extrapolate_trend='freq')
        fig_x = decomposition_x.plot()
        fig_x.set_size_inches(14,7)
        plt.show()
        st.pyplot(fig_x)
        
    if chart_visual_seasonal_decomposition == "Need of resources":
        seasonal_decompose(y)
    elif chart_visual_seasonal_decomposition == "Availability of resources":
        seasonal_decompose(z)

    
    
    

elif control == 'XAI':
    ### XAI - Explainable Artificial Intelligence
    # Dataset
    # Load Dataset
    tweets = pd.read_csv("dataset.csv",lineterminator='\n')[["text","type","dataset\r"]]
    tweets.columns = ["text","type", "dataset"]
    tweets['dataset'] = tweets['dataset'].str.replace(r'\r', '')
    # Selection of Input & Output Variables
    X = tweets.loc[:, 'text']
    Y = tweets.loc[:, 'type']
    X = list(X)
    
    
    def preprocess_dataset(d):    
        # Define count variables
        cnt=0
        punctuation_count = 0
        digit_count = 0
            
        # Convert the corpus to lowercase
        lower_corpus = []
        for i in range(len(d)):
            lower_corpus.append(" ".join([word.lower() for word in d[i].split()]))
                        
        # Remove any special symbol or punctuation
        without_punctuation_corpus = []
        for i in range(len(lower_corpus)):
            p = []
            for ch in lower_corpus[i]:
                if ch not in string.punctuation:
                    p.append(ch)
                else:
                    p.append(" ")
                    # Count of punctuation marks removed
                    punctuation_count += 1
            x = ''.join(p)
            if len(x) > 0:  
                without_punctuation_corpus.append(x)
          
        # Remove urls with http, https or www and Retweets RT
        without_url_corpus = []
        for i in range(len(without_punctuation_corpus)):
            text = without_punctuation_corpus[i]
            text = re.sub(r"http\S*||www\S*", "", text)
            text = re.sub(r"RT ", "", text)
            without_url_corpus.append(text)
            
        # Remove special characters and numbers from the corpus
        without_digit_corpus = []
        for i in range(len(without_url_corpus)):
            p = []
            for word in without_url_corpus[i].split():
                if word.isalpha():
                    p.append(word)
                else:
                    # Count of punctuation marks removed
                    digit_count += 1
            x = ' '.join(p)
            without_digit_corpus.append(x)
            
                
        # Tokenize the corpus
        # word_tokenize(s): Tokenize a string to split off punctuation other than periods
        # With the help of nltk.tokenize.word_tokenize() method, we are able to extract the tokens
        # from string of characters by using tokenize.word_tokenize() method. 
        # Tokenization was done to support efficient removal of stopwords
        total_count = 0
        tokenized_corpus = []
        for i in without_digit_corpus:
            tokenized_tweet = nltk.word_tokenize(i)
            tokenized_corpus.append(tokenized_tweet)
            # Count the length of tokenized corpus
            total_count += len(list(tokenized_tweet))
        
        
        # Remove Stopwords
        stopw = stopwords.words('english')
        count = 0
        tokenized_corpus_no_stopwords = []  
        for i,c in enumerate(tokenized_corpus): 
            tokenized_corpus_no_stopwords.append([])
            for word in c: 
                if word not in stopw:  
                    tokenized_corpus_no_stopwords[i].append(word) 
                else:
                    count += 1
    
        # lemmatization and removing words that are too large and small
        lemmatized_corpus = []
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        ct = 0
        cnt_final=0
        dictt = {}
        for i in range(0,len(tokenized_corpus_no_stopwords)):
            lemmatized_corpus.append([])
            for w in tokenized_corpus_no_stopwords[i]:
                # lematizing only those words whose length >= 2 and <=10
                # Considering words with length greater than or equal to 2 and less than or equal to 10
                if(len(w)>2 and len(w)<=10):
                    lemmatized_corpus[i].append(lemmatizer.lemmatize(w))
                    cnt_final+=1
                # Count of final corpus
                # This is the length of total corpus that went through the process of lematization
                ct+=1
      
        ############## Removing words of large and small length
        # Doing a survey to find out the length of words so we can remove the too small and too large words from the Corpus
        # plt.bar(*zip(*dictt.items()))
        # plt.show()
    
        # Punctuation Preprocessing
        preprocessed_corpus = []
        for i,c in enumerate(lemmatized_corpus):
            preprocessed_corpus.append([])
            for word in c:
                x = ''.join([ch for ch in word if ch not in string.punctuation])
                if len(x) > 0:
                    preprocessed_corpus[i].append(x)
            
       
        # Clear unwanted data variables to save RAM due to memory limitations
        del lower_corpus
        del without_punctuation_corpus
        del without_digit_corpus
        del tokenized_corpus
        del tokenized_corpus_no_stopwords
        del lemmatized_corpus
        return preprocessed_corpus
    
    # Preprocess the Input Variables
    preprocessed_corpus = preprocess_dataset(X)
    
    data_corpus = []
    for i in preprocessed_corpus:
        data_corpus.append(" ".join([w for w in i]))
        
    # Vectorization of Preprocessed Tweets
    
    tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words='english',min_df=2, max_features=50, ngram_range = (1,3))
    X_tfidf = tfidfvectorizer.fit_transform(data_corpus)
    
    # Feature extraction
    feature_names = tfidfvectorizer.get_feature_names()
    
    # Splitting the input into test and train
    X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf.toarray(), Y, train_size = 0.7)
    
    # intializing the model
    model = LGBMClassifier()
    model.fit(X_train,Y_train)
    
    
    # Instantiating the explainer object by passing in the training set, and the extracted features
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train,feature_names=feature_names,verbose=True, mode='classification', class_names=[0,1,2])
    # Streamlit Code starts here
    st.title('XAI - Explainable Artificial Intelligence')
    st.markdown("The dashboard will help the users verify the efficiency of the classification model used here")
    
    st.sidebar.markdown('<h2 style="color:#E07C24;">Class Labels</h2>',unsafe_allow_html=True)
    st.sidebar.markdown('<h3 style="color:#8D3DAF;">0 represents Need</h3>', unsafe_allow_html=True)
    st.sidebar.markdown('<h3 style="color:#8D3DAF;">1 represents Availability</h3>', unsafe_allow_html=True)
    st.sidebar.markdown('<h3 style="color:#8D3DAF;">2 represents Others</h3>', unsafe_allow_html=True)
       
    h = st.slider('Select the Tweet using the slider', 0, len(X)-1, 18)       
    idx=0 # the rows of the dataset
    explainable_exp = explainer_lime.explain_instance(X_tfidf.toarray()[h], model.predict_proba, num_features=10, labels=[0,1,2])
    #explainable_exp.show_in_notebook(show_table=True, show_all=False)
    html = explainable_exp.as_html()
    
    
    st.write('**Tweet:**', X[h])
    st.write('**Label:**', Y[h])
    
    components.html(html, height=800)

    
    
    
    

    
# Footer
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

footer {visibility: hidden;}

.footer {
margin:0;
height:5px;
position:relative;
top:140px;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with <span style='color:red;'>❤</span> by <a style='text-align: center;' href="https://github.com/26aseem" target="_blank">Team CPG-7</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)