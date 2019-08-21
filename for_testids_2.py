# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:35:04 2019

@author: Administrator
"""
from selenium import webdriver
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from time import sleep
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
#import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()
#from googletrans import Translator
#translator = Translator()
import json
#import nltk
import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#nltk.download('stopwords')
import os,sys
import operator
import Cleaning
#from Cleaning import translatepkg
import translatescrp
#import pickle
from sklearn.externals import joblib
from random import randint
import requests
os.getcwd()
import flask
from flask import request, jsonify
import pyperclip
import sklearn

import pandas as pd
import requests
import json
import math
import urllib
import pyperclip
import random
import emoji
emoji_pattern1= emoji.get_emoji_regexp()
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from random import randint
from selenium.webdriver.support.ui import Select
import time

os.chdir("C:\Server backup\VIDOOLY NILANJAN\Python\Translate\For 34")

def desc(category):
    switcher = {
                "Adult":"Adult and explicit content",
                "Agriculture, Farming & Gardening":"farming, gardening, animal husbandary and agricultural related videos including goat farming, buffallo farming, fish farming etc",
                "Art & Craft":"Art and crafting related videos",
                "Astrology":"Astrology, horoscope and jyotish related videos",
                "Auto & Vehicles":"Bike, cars, locomotive related reviews, tutorials and first looks etc.,",
                "Beauty":"Beauty and makeup related tutorials, unboxing and reviews, Mehindi, hair, nail related videos",
                "Business & Finance":"Banking, finance and insurance related videos",
                "Comedy":"Funny, sketch, vine, spoof comedy related videos including funny animals",
                "Education":"Coaching, exam preparation and education related content",
                "Entertainment":"Movie clips, films, trailers and teasers, audio launches, entertainment news, short films, TV daily, whatsapp status, relationship videos etc.",
                "Family & Parenting":"Baby care, parenting tips, food for babies",
                "Fashion & Style":"clothing, fashion and apparel related videos",
                "Food & Recipe":"Food preparation, food recipes and food related tutorials",
                "Gaming":"Gameplay, game tutorials, gaming glitches, crashes, how to play games, train simulator, puzzles, quizzes and riddles",
                "Health & Fitness":"Fitness tutorials, fitness diet, health information, health tips etc.,",
                "Information":"videos providing information on any topic, business ideas, e commerce, info on paytm etc., relationships, lifehacks, tips, videos on chanakya neeti, paytm, google tez, phone pe related content",
                "Infotainment":"videos providing information as well as entertainment such as discovery and national geo videos, mysteries, stories, facts, magic and biography, videos on aliens, space and universe",
                "IT & Services":"Software related content such as tutorials on software",
                "Kids & Animation":"kids rhymes, poems, kids songs, kids animation etc.,",
                "Motivational":"motivational, inspirational speeches, talks etc.,",
                "Music & Dance":"audio songs, video songs, lyrical songs, dance performances, dance tutorials, devotional songs, Bhajan, Kirtan, Hymn, Qawali, Ghazal, Shayari",
                "News & Politics":"All news related contents ( except entertainment news), news about politician",
                "People & Culture":"Videos on culture, festivals, celebrations of a particular festival etc., including mahabharata and bhagavad gita related content",
                "Pets & Animals":"Videos on animals, animal care, pet care etc.",
                "Photography":"Photography, video shooting and editing tips, tutorials, official photography channels etc.",
                "Real Estate & Home Decor":"Real estate, construction, building home tips etc., home decor, furniture",
                "Religious":"religious and spiritual contents, speeches, talks, bhajans etc.",
                "Science & Technology":"videos related to science & technology like tips, reviews, unboxing on mobiles, laptops etc.",
                "Sports":"All kinds of sports videos like, sports highlights, tutorial, training, tips and tricks etc.",
                "Supernatural":"real ghost stories, supernatural content, Tantrik, Aghori, horror, scary related videos",
                "Travel & Leisure":"travel locations, travel expenditure, travelling tips etc.",
                "Vlogging":"Any video Blogging",
                "NA":"Not Available"
                }
    return switcher.get(category)

filename = 'finalized_model_final.sav'
filename1 = 'tfidf_final.sav'
filename2 = 'category_to_id_final.sav'
filename3 = 'id_to_category_final.sav'
filename4 = "Min_thresh_final.sav"

loaded_model = joblib.load(open(filename, 'rb'))
loaded_tfidf = joblib.load(open(filename1, 'rb'))
loaded_category_to_id = joblib.load(open(filename2, 'rb'))
loaded_id_to_category = joblib.load(open(filename3, 'rb'))
min_thresh = joblib.load(open(filename4, 'rb'))

# =============================================================================
driver = webdriver.Chrome('C:\Server backup\VIDOOLY NILANJAN\Python\Selenium Drivers\chromedriver_win32\chromedriver\chromedriver.exe') 
driver.minimize_window()

driver.minimize_window()

driver.get('https://translate.google.com/')
print ("Opened Site")
sleep(3)
username_box = driver.find_element_by_xpath('//*[@id="source"]')
 #    username_box.send_keys("hello")
username_box.send_keys(u"हिंदी")

sleep(3)
login_box = driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[3]/div[1]/div[4]/div[4]/div')
#login_box = driver.find_element_by_xpath('//*[@id="gt-res-copy"]/span')
clearbox=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div/div/div[2]/div[1]/div/div')
# =============================================================================
#transbox=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div/div/div[2]/div[2]/div/div/div')

#/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[3]/div[1]/div[4]/div[4]/div

def translatescrp1(df):
    file3 = pd.DataFrame(index=range(len(df.index)), columns=["id","Meta","Transmeta"])
    file3 = file3.fillna("")
    
    for i in range(len(df.index)):
        file3.iloc[i,0]=df.iloc[i,0]
        file3.iloc[i,1]=df.iloc[i,1]
        
        #    username_box.send_keys(u"हिंदी")
        counter=0
        
        def split(input,size):
            return [input[start:start+size] for start in range(0,len(input),size)]
        #    tmptxt1=range(len(text10))
        #    stop_words=pd.read_csv('C:/Users/Administrator/Desktop/VIDOOLY NILANJAN/Python/Translate/stop_words.csv',encoding='utf-8',header=None)
        #    stop_word=[]
        #    for i in range(len(stop_words)):
        #        stop_word.append(stop_words.iloc[i,0])
            
        #    counter=0
        #    for i in range(len(text10)):
        trans=""
        meta=df.iloc[i,1]
#        meta="Puneeth Rajkumar|ಪುನೀತ್ ರಾಜಕುಮಾರ್ ಅವರಿಗೆ ದೊಡ್ಡ ಸಂಕಷ್ಟ! cinemantra24 ಪುನೀತ್ ರಾಜಕುಮಾರ್ ಅವರಿಗೆ ದೊಡ್ಡ ಸಂಕಷ್ಟ!"
        
        #    meta=file1.iloc[6,1]
        text1= meta.replace('🏻','').replace('_','').replace('`','').replace('“','').replace('”','').replace('$','').replace('¢','').replace('£','').replace('¤','').replace('¥','').replace('֏','').replace('؋','').replace('৲','').replace('৳','').replace('৻','').replace('૱','').replace('௹','').replace('฿','').replace('៛','').replace('₠','').replace('₡','').replace('₢','').replace('₣','').replace('₤','').replace('₥','').replace('₦','').replace('₧','').replace('₨','').replace('₩','').replace('₪','').replace('₫','').replace('€','').replace('₭','').replace('₮','').replace('₯','').replace('₰','').replace('₱','').replace('₲','').replace('₳','').replace('₴','').replace('₵','').replace('₶','').replace('₷','').replace('₸','').replace('₹','').replace('₺','').replace('₻','').replace('₼','').replace('₽','').replace('₾','').replace('₿','').replace('﷼','').replace('﹩','').replace('＄','').replace('￠','').replace('￡','').replace('￥','').replace('￦','').replace("&","").replace("'","").replace("°","").replace("’","").replace("🥇","").replace("🥈","").replace("🥉","").replace("🆎","").replace("🏧","").replace("🅰","").replace("🇦🇫","").replace("🇦🇱","").replace("🇩🇿","").replace("🇦🇸","").replace("🇦🇩","").replace("🇦🇴","").replace("🇦🇮","").replace("🇦🇶","").replace("🇦🇬","").replace("♒","").replace("🇦🇷","").replace("♈","").replace("🇦🇲","").replace("🇦🇼","").replace("🇦🇨","").replace("🇦🇺","").replace("🇦🇹","").replace("🇦🇿","").replace("🔙","").replace("🅱","").replace("🇧🇸","").replace("🇧🇭","").replace("🇧🇩","").replace("🇧🇧","").replace("🇧🇾","").replace("🇧🇪","").replace("🇧🇿","").replace("🇧🇯","").replace("🇧🇲","").replace("🇧🇹","").replace("🇧🇴","").replace("🇧🇦","").replace("🇧🇼","").replace("🇧🇻","").replace("🇧🇷","").replace("🇮🇴","").replace("🇻🇬","").replace("🇧🇳","").replace("🇧🇬","").replace("🇧🇫","").replace("🇧🇮","").replace("🆑","").replace("🆒","").replace("🇰🇭","").replace("🇨🇲","").replace("🇨🇦","").replace("🇮🇨","").replace("♋","").replace("🇨🇻","").replace("♑","").replace("🇧🇶","").replace("🇰🇾","").replace("🇨🇫","").replace("🇪🇦","").replace("🇹🇩","").replace("🇨🇱","").replace("🇨🇳","").replace("🇨🇽","").replace("🎄","").replace("🇨🇵","").replace("🇨🇨","").replace("🇨🇴","").replace("🇰🇲","").replace("🇨🇬","").replace("🇨🇩","").replace("🇨🇰","").replace("🇨🇷","").replace("🇭🇷","").replace("🇨🇺","").replace("🇨🇼","").replace("🇨🇾","").replace("🇨🇿","").replace("🇨🇮","").replace("🇩🇰","").replace("🇩🇬","").replace("🇩🇯","").replace("🇩🇲","").replace("🇩🇴","").replace("🔚","").replace("🇪🇨","").replace("🇪🇬","").replace("🇸🇻","").replace("🏴󠁧󠁢󠁥󠁮󠁧󠁿","").replace("🇬🇶","").replace("🇪🇷","").replace("🇪🇪","").replace("🇪🇹","").replace("🇪🇺","").replace("🆓","").replace("🇫🇰","").replace("🇫🇴","").replace("🇫🇯","").replace("🇫🇮","").replace("🇫🇷","").replace("🇬🇫","").replace("🇵🇫","").replace("🇹🇫","").replace("🇬🇦","").replace("🇬🇲","").replace("♊","").replace("🇬🇪","").replace("🇩🇪","").replace("🇬🇭","").replace("🇬🇮","").replace("🇬🇷","").replace("🇬🇱","").replace("🇬🇩","").replace("🇬🇵","").replace("🇬🇺","").replace("🇬🇹","").replace("🇬🇬","").replace("🇬🇳","").replace("🇬🇼","").replace("🇬🇾","").replace("🇭🇹","").replace("🇭🇲","").replace("🇭🇳","").replace("🇭🇰","").replace("🇭🇺","").replace("🆔","").replace("🇮🇸","").replace("🇮🇳","").replace("🇮🇩","").replace("🇮🇷","").replace("🇮🇶","").replace("🇮🇪","").replace("🇮🇲","").replace("🇮🇱","").replace("🇮🇹","").replace("🇯🇲","").replace("🇯🇵","").replace("🉑","").replace("🈸","").replace("🉐","").replace("🏯","").replace("㊗","").replace("🈹","").replace("🎎","").replace("🈚","").replace("🈁","").replace("🈷","").replace("🈵","").replace("🈶","").replace("🈺","").replace("🈴","").replace("🏣","").replace("🈲","").replace("🈯","").replace("㊙","").replace("🈂","").replace("🔰","").replace("🈳","").replace("🇯🇪","").replace("🇯🇴","").replace("🇰🇿","").replace("🇰🇪","").replace("🇰🇮","").replace("🇽🇰","").replace("🇰🇼","").replace("🇰🇬","").replace("🇱🇦","").replace("🇱🇻","").replace("🇱🇧","").replace("♌","").replace("🇱🇸","").replace("🇱🇷","").replace("♎","").replace("🇱🇾","").replace("🇱🇮","").replace("🇱🇹","").replace("🇱🇺","").replace("🇲🇴","").replace("🇲🇰","").replace("🇲🇬","").replace("🇲🇼","").replace("🇲🇾","").replace("🇲🇻","").replace("🇲🇱","").replace("🇲🇹","").replace("🇲🇭","").replace("🇲🇶","").replace("🇲🇷","").replace("🇲🇺","").replace("🇾🇹","").replace("🇲🇽","").replace("🇫🇲","").replace("🇲🇩","").replace("🇲🇨","").replace("🇲🇳","").replace("🇲🇪","").replace("🇲🇸","").replace("🇲🇦","").replace("🇲🇿","").replace("🤶","").replace("🤶🏿","").replace("🤶🏻","").replace("🤶🏾","").replace("🤶🏼","").replace("🤶🏽","").replace("🇲🇲","").replace("🆕","").replace("🆖","").replace("🇳🇦","").replace("🇳🇷","").replace("🇳🇵","").replace("🇳🇱","").replace("🇳🇨","").replace("🇳🇿","").replace("🇳🇮","").replace("🇳🇪","").replace("🇳🇬","").replace("🇳🇺","").replace("🇳🇫","").replace("🇰🇵","").replace("🇲🇵","").replace("🇳🇴","").replace("🆗","").replace("👌","").replace("👌🏿","").replace("👌🏻","").replace("👌🏾","").replace("👌🏼","").replace("👌🏽","").replace("🔛","").replace("🅾","").replace("🇴🇲","").replace("⛎","").replace("🅿","").replace("🇵🇰","").replace("🇵🇼","").replace("🇵🇸","").replace("🇵🇦","").replace("🇵🇬","").replace("🇵🇾","").replace("🇵🇪","").replace("🇵🇭","").replace("♓","").replace("🇵🇳","").replace("🇵🇱","").replace("🇵🇹","").replace("🇵🇷","").replace("🇶🇦","").replace("🇷🇴","").replace("🇷🇺","").replace("🇷🇼","").replace("🇷🇪","").replace("🔜","").replace("🆘","").replace("♐","").replace("🇼🇸","").replace("🇸🇲","").replace("🎅","").replace("🎅🏿","").replace("🎅🏻","").replace("🎅🏾","").replace("🎅🏼","").replace("🎅🏽","").replace("🇸🇦","").replace("♏","").replace("🏴󠁧󠁢󠁳󠁣󠁴󠁿","").replace("🇸🇳","").replace("🇷🇸","").replace("🇸🇨","").replace("🇸🇱","").replace("🇸🇬","").replace("🇸🇽","").replace("🇸🇰","").replace("🇸🇮","").replace("🇸🇧","").replace("🇸🇴","").replace("🇿🇦","").replace("🇬🇸","").replace("🇰🇷","").replace("🇸🇸","").replace("🇪🇸","").replace("🇱🇰","").replace("🇧🇱","").replace("🇸🇭","").replace("🇰🇳","").replace("🇱🇨","").replace("🇲🇫","").replace("🇵🇲","").replace("🇻🇨","").replace("🗽","").replace("🇸🇩","").replace("🇸🇷","").replace("🇸🇯","").replace("🇸🇿","").replace("🇸🇪","").replace("🇨🇭","").replace("🇸🇾","").replace("🇸🇹","").replace("🦖","").replace("🔝","").replace("🇹🇼","").replace("🇹🇯","").replace("🇹🇿","").replace("♉","").replace("🇹🇭","").replace("🇹🇱","").replace("🇹🇬","").replace("🇹🇰","").replace("🗼","").replace("🇹🇴","").replace("🇹🇹","").replace("🇹🇦","").replace("🇹🇳","").replace("🇹🇷","").replace("🇹🇲","").replace("🇹🇨","").replace("🇹🇻","").replace("🇺🇲","").replace("🇻🇮","").replace("🆙","").replace("🇺🇬","").replace("🇺🇦","").replace("🇦🇪","").replace("🇬🇧","").replace("🇺🇳","").replace("🇺🇸","").replace("🇺🇾","").replace("🇺🇿","").replace("🆚","").replace("🇻🇺","").replace("🇻🇦","").replace("🇻🇪","").replace("🇻🇳","").replace("♍","").replace("🏴󠁧󠁢󠁷󠁬󠁳󠁿","").replace("🇼🇫","").replace("🇪🇭","").replace("🇾🇪","").replace("🇿🇲","").replace("🇿🇼","").replace("🎟","").replace("🧑","").replace("🧑🏿","").replace("🧑🏻","").replace("🧑🏾","").replace("🧑🏼","").replace("🧑🏽","").replace("🚡","").replace("✈","").replace("🛬","").replace("🛫","").replace("⏰","").replace("⚗","").replace("👽","").replace("👾","").replace("🚑","").replace("🏈","").replace("🏺","").replace("⚓","").replace("💢","").replace("😠","").replace("👿","").replace("😧","").replace("🐜","").replace("📶","").replace("😰","").replace("🚛","").replace("🎨","").replace("😲","").replace("⚛","").replace("🚗","").replace("🥑","").replace("👶","").replace("👼","").replace("👼🏿","").replace("👼🏻","").replace("👼🏾","").replace("👼🏼","").replace("👼🏽","").replace("🍼","").replace("🐤","").replace("👶🏿","").replace("👶🏻","").replace("👶🏾","").replace("👶🏼","").replace("👶🏽","").replace("🚼","").replace("👇","").replace("👇🏿","").replace("👇🏻","").replace("👇🏾","").replace("👇🏼","").replace("👇🏽","").replace("👈","").replace("👈🏿","").replace("👈🏻","").replace("👈🏾","").replace("👈🏼","").replace("👈🏽","").replace("👉","").replace("👉🏿","").replace("👉🏻","").replace("👉🏾","").replace("👉🏼","").replace("👉🏽","").replace("👆","").replace("👆🏿","").replace("👆🏻","").replace("👆🏾","").replace("👆🏼","").replace("👆🏽","").replace("🥓","").replace("🏸","").replace("🛄","").replace("🥖","").replace("⚖","").replace("🎈","").replace("🗳","").replace("☑","").replace("🍌","").replace("🏦","").replace("📊","").replace("💈","").replace("⚾","").replace("🏀","").replace("🦇","").replace("🛁","").replace("🔋","").replace("🏖","").replace("😁","").replace("🐻","").replace("🧔","").replace("🧔🏿","").replace("🧔🏻","").replace("🧔🏾","").replace("🧔🏼","").replace("🧔🏽","").replace("💓","").replace("🛏","").replace("🍺","").replace("🔔","").replace("🔕","").replace("🛎","").replace("🍱","").replace("🚲","").replace("👙","").replace("🧢","").replace("☣","").replace("🐦","").replace("🎂","").replace("⚫","").replace("🏴","").replace("🖤","").replace("⬛","").replace("◾","").replace("◼","").replace("✒","").replace("▪","").replace("🔲","").replace("👱‍♂️","").replace("👱🏿‍♂️","").replace("👱🏻‍♂️","").replace("👱🏾‍♂️","").replace("👱🏼‍♂️","").replace("👱🏽‍♂️","").replace("👱","").replace("👱🏿","").replace("👱🏻","").replace("👱🏾","").replace("👱🏼","").replace("👱🏽","").replace("👱‍♀️","").replace("👱🏿‍♀️","").replace("👱🏻‍♀️","").replace("👱🏾‍♀️","").replace("👱🏼‍♀️","").replace("👱🏽‍♀️","").replace("🌼","").replace("🐡","").replace("📘","").replace("🔵","").replace("💙","").replace("🐗","").replace("💣","").replace("🔖","").replace("📑","").replace("📚","").replace("🍾","").replace("💐","").replace("🏹","").replace("🥣","").replace("🎳","").replace("🥊","").replace("👦","").replace("👦🏿","").replace("👦🏻","").replace("👦🏾","").replace("👦🏼","").replace("👦🏽","").replace("🧠","").replace("🍞","").replace("🤱","").replace("🤱🏿","").replace("🤱🏻","").replace("🤱🏾","").replace("🤱🏼","").replace("🤱🏽","").replace("👰","").replace("👰🏿","").replace("👰🏻","").replace("👰🏾","").replace("👰🏼","").replace("👰🏽","").replace("🌉","").replace("💼","").replace("🔆","").replace("🥦","").replace("💔","").replace("🐛","").replace("🏗","").replace("🚅","").replace("🌯","").replace("🚌","").replace("🚏","").replace("👤","").replace("👥","").replace("🦋","").replace("🌵","").replace("📆","").replace("🤙","").replace("🤙🏿","").replace("🤙🏻","").replace("🤙🏾","").replace("🤙🏼","").replace("🤙🏽","").replace("🐫","").replace("📷","").replace("📸","").replace("🏕","").replace("🕯","").replace("🍬","").replace("🥫","").replace("🛶","").replace("🗃","").replace("📇","").replace("🗂","").replace("🎠","").replace("🎏","").replace("🥕","").replace("🏰","").replace("🐱","").replace("🐱","").replace("😹","").replace("😼","").replace("⛓","").replace("📉","").replace("📈","").replace("💹","").replace("🧀","").replace("🏁","").replace("🍒","").replace("🌸","").replace("🌰","").replace("🐔","").replace("🧒","").replace("🧒🏿","").replace("🧒🏻","").replace("🧒🏾","").replace("🧒🏼","").replace("🧒🏽","").replace("🚸","").replace("🐿","").replace("🍫","").replace("🥢","").replace("⛪","").replace("🚬","").replace("🎦","").replace("Ⓜ","").replace("🎪","").replace("🏙","").replace("🌆","").replace("🗜","").replace("🎬","").replace("👏","").replace("👏🏿","").replace("👏🏻","").replace("👏🏾","").replace("👏🏼","").replace("👏🏽","").replace("🏛","").replace("🍻","").replace("🥂","").replace("📋","").replace("🔃","").replace("📕","").replace("📪","").replace("📫","").replace("🌂","").replace("☁","").replace("🌩","").replace("⛈","").replace("🌧","").replace("🌨","").replace("🤡","").replace("♣","").replace("👝","").replace("🧥","").replace("🍸","").replace("🥥","").replace("⚰","").replace("💥","").replace("☄","").replace("💽","").replace("🖱","").replace("🎊","").replace("😖","").replace("😕","").replace("🚧","").replace("👷","").replace("👷🏿","").replace("👷🏻","").replace("👷🏾","").replace("👷🏼","").replace("👷🏽","").replace("🎛","").replace("🏪","").replace("🍚","").replace("🍪","").replace("🍳","").replace("©","").replace("🛋","").replace("🔄","").replace("💑","").replace("👨‍❤️‍👨","").replace("👩‍❤️‍👨","").replace("👩‍❤️‍👩","").replace("🐮","").replace("🐮","").replace("🤠","").replace("🦀","").replace("🖍","").replace("💳","").replace("🌙","").replace("🦗","").replace("🏏","").replace("🐊","").replace("🥐","").replace("❌","").replace("❎","").replace("🤞","").replace("🤞🏿","").replace("🤞🏻","").replace("🤞🏾","").replace("🤞🏼","").replace("🤞🏽","").replace("🎌","").replace("⚔","").replace("👑","").replace("😿","").replace("😢","").replace("🔮","").replace("🥒","").replace("🥤","").replace("🥌","").replace("➰","").replace("💱","").replace("🍛","").replace("🍮","").replace("🛃","").replace("🥩","").replace("🌀","").replace("🗡","").replace("🍡","").replace("💨","").replace("🌳","").replace("🦌","").replace("🚚","").replace("🏬","").replace("🏚","").replace("🏜","").replace("🏝","").replace("🖥","").replace("🕵","").replace("🕵🏿","").replace("🕵🏻","").replace("🕵🏾","").replace("🕵🏼","").replace("🕵🏽","").replace("♦","").replace("💠","").replace("🔅","").replace("🎯","").replace("😞","").replace("💫","").replace("😵","").replace("🐶","").replace("🐶","").replace("💵","").replace("🐬","").replace("🚪","").replace("🔯","").replace("➿","").replace("‼","").replace("🍩","").replace("🕊","").replace("↙","").replace("↘","").replace("⬇","").replace("😓","").replace("🔽","").replace("🐉","").replace("🐲","").replace("👗","").replace("🤤","").replace("💧","").replace("🥁","").replace("🦆","").replace("🥟","").replace("📀","").replace("📧","").replace("🦅","").replace("👂","").replace("👂🏿","").replace("👂🏻","").replace("👂🏾","").replace("👂🏼","").replace("👂🏽","").replace("🌽","").replace("🍳","").replace("🍆","").replace("✴","").replace("✳","").replace("🕣","").replace("🕗","").replace("⏏","").replace("🔌","").replace("🐘","").replace("🕦","").replace("🕚","").replace("🧝","").replace("🧝🏿","").replace("🧝🏻","").replace("🧝🏾","").replace("🧝🏼","").replace("🧝🏽","").replace("✉","").replace("📩","").replace("💶","").replace("🌲","").replace("🐑","").replace("❗","").replace("⁉","").replace("🤯","").replace("😑","").replace("👁","").replace("👁️‍🗨️","").replace("👀","").replace("😘","").replace("😋","").replace("😱","").replace("🤮","").replace("🤭","").replace("🤕","").replace("😷","").replace("🧐","").replace("😮","").replace("🤨","").replace("🙄","").replace("😤","").replace("🤬","").replace("😂","").replace("🤒","").replace("😛","").replace("😶","").replace("🏭","").replace("🧚","").replace("🧚🏿","").replace("🧚🏻","").replace("🧚🏾","").replace("🧚🏼","").replace("🧚🏽","").replace("🍂","").replace("👪","").replace("👨‍👦","").replace("👨‍👦‍👦","").replace("👨‍👧","").replace("👨‍👧‍👦","").replace("👨‍👧‍👧","").replace("👨‍👨‍👦","").replace("👨‍👨‍👦‍👦","").replace("👨‍👨‍👧","").replace("👨‍👨‍👧‍👦","").replace("👨‍👨‍👧‍👧","").replace("👨‍👩‍👦","").replace("👨‍👩‍👦‍👦","").replace("👨‍👩‍👧","").replace("👨‍👩‍👧‍👦","").replace("👨‍👩‍👧‍👧","").replace("👩‍👦","").replace("👩‍👦‍👦","").replace("👩‍👧","").replace("👩‍👧‍👦","").replace("👩‍👧‍👧","").replace("👩‍👩‍👦","").replace("👩‍👩‍👦‍👦","").replace("👩‍👩‍👧","").replace("👩‍👩‍👧‍👦","").replace("👩‍👩‍👧‍👧","").replace("⏩","").replace("⏬","").replace("⏪","").replace("⏫","").replace("📠","").replace("😨","").replace("♀","").replace("🎡","").replace("⛴","").replace("🏑","").replace("🗄","").replace("📁","").replace("🎞","").replace("📽","").replace("🔥","").replace("🚒","").replace("🎆","").replace("🌓","").replace("🌛","").replace("🐟","").replace("🍥","").replace("🎣","").replace("🕠","").replace("🕔","").replace("⛳","").replace("🔦","").replace("⚜","").replace("💪","").replace("💪🏿","").replace("💪🏻","").replace("💪🏾","").replace("💪🏼","").replace("💪🏽","").replace("💾","").replace("🎴","").replace("😳","").replace("🛸","").replace("🌫","").replace("🌁","").replace("🙏","").replace("🙏🏿","").replace("🙏🏻","").replace("🙏🏾","").replace("🙏🏼","").replace("🙏🏽","").replace("👣","").replace("🍴","").replace("🍽","").replace("🥠","").replace("⛲","").replace("🖋","").replace("🕟","").replace("🍀","").replace("🕓","").replace("🦊","").replace("🖼","").replace("🍟","").replace("🍤","").replace("🐸","").replace("🐥","").replace("☹","").replace("😦","").replace("⛽","").replace("🌕","").replace("🌝","").replace("⚱","").replace("🎲","").replace("⚙","").replace("💎","").replace("🧞","").replace("👻","").replace("🦒","").replace("👧","").replace("👧🏿","").replace("👧🏻","").replace("👧🏾","").replace("👧🏼","").replace("👧🏽","").replace("🥛","").replace("👓","").replace("🌎","").replace("🌏","").replace("🌍","").replace("🌐","").replace("🧤","").replace("🌟","").replace("🥅","").replace("🐐","").replace("👺","").replace("🦍","").replace("🎓","").replace("🍇","").replace("🍏","").replace("📗","").replace("💚","").replace("🥗","").replace("😬","").replace("😺","").replace("😸","").replace("😀","").replace("😃","").replace("😄","").replace("😅","").replace("😆","").replace("💗","").replace("💂","").replace("💂🏿","").replace("💂🏻","").replace("💂🏾","").replace("💂🏼","").replace("💂🏽","").replace("🎸","").replace("🍔","").replace("🔨","").replace("⚒","").replace("🛠","").replace("🐹","").replace("🖐","").replace("🖐🏿","").replace("🖐🏻","").replace("🖐🏾","").replace("🖐🏼","").replace("🖐🏽","").replace("👜","").replace("🤝","").replace("🐣","").replace("🎧","").replace("🙉","").replace("💟","").replace("♥","").replace("💘","").replace("💝","").replace("✔","").replace("➗","").replace("💲","").replace("❣","").replace("⭕","").replace("➖","").replace("✖","").replace("➕","").replace("🦔","").replace("🚁","").replace("🌿","").replace("🌺","").replace("👠","").replace("🚄","").replace("⚡","").replace("🕳","").replace("🍯","").replace("🐝","").replace("🚥","").replace("🐴","").replace("🐴","").replace("🏇","").replace("🏇🏿","").replace("🏇🏻","").replace("🏇🏾","").replace("🏇🏼","").replace("🏇🏽","").replace("🏥","").replace("☕","").replace("🌭","").replace("🌶","").replace("♨","").replace("🏨","").replace("⌛","").replace("⏳","").replace("🏠","").replace("🏡","").replace("🏘","").replace("🤗","").replace("💯","").replace("😯","").replace("🍨","").replace("🏒","").replace("⛸","").replace("📥","").replace("📨","").replace("☝","").replace("☝🏿","").replace("☝🏻","").replace("☝🏾","").replace("☝🏼","").replace("☝🏽","").replace("ℹ","").replace("🔤","").replace("🔡","").replace("🔠","").replace("🔢","").replace("🔣","").replace("🎃","").replace("👖","").replace("🃏","").replace("🕹","").replace("🕋","").replace("🔑","").replace("⌨","").replace("#️⃣","").replace("*️⃣","").replace("0️⃣","").replace("1️⃣","").replace("🔟","").replace("2️⃣","").replace("3️⃣","").replace("4️⃣","").replace("5️⃣","").replace("6️⃣","").replace("7️⃣","").replace("8️⃣","").replace("9️⃣","").replace("🛴","").replace("👘","").replace("💋","").replace("👨‍❤️‍💋‍👨","").replace("💋","").replace("👩‍❤️‍💋‍👨","").replace("👩‍❤️‍💋‍👩","").replace("😽","").replace("😗","").replace("😚","").replace("😙","").replace("🔪","").replace("🥝","").replace("🐨","").replace("🏷","").replace("🐞","").replace("💻","").replace("🔷","").replace("🔶","").replace("🌗","").replace("🌜","").replace("⏮","").replace("✝","").replace("🍃","").replace("📒","").replace("🤛","").replace("🤛🏿","").replace("🤛🏻","").replace("🤛🏾","").replace("🤛🏼","").replace("🤛🏽","").replace("↔","").replace("⬅","").replace("↪","").replace("🛅","").replace("🗨","").replace("🍋","").replace("🐆","").replace("🎚","").replace("💡","").replace("🚈","").replace("🔗","").replace("🖇","").replace("🦁","").replace("💄","").replace("🚮","").replace("🦎","").replace("🔒","").replace("🔐","").replace("🔏","").replace("🚂","").replace("🍭","").replace("😭","").replace("📢","").replace("🤟","").replace("🤟🏿","").replace("🤟🏻","").replace("🤟🏾","").replace("🤟🏼","").replace("🤟🏽","").replace("🏩","").replace("💌","").replace("🤥","").replace("🧙","").replace("🧙🏿","").replace("🧙🏻","").replace("🧙🏾","").replace("🧙🏼","").replace("🧙🏽","").replace("🔍","").replace("🔎","").replace("🀄","").replace("♂","").replace("👨","").replace("👫","").replace("👨‍🎨","").replace("👨🏿‍🎨","").replace("👨🏻‍🎨","").replace("👨🏾‍🎨","").replace("👨🏼‍🎨","").replace("👨🏽‍🎨","").replace("👨‍🚀","").replace("👨🏿‍🚀","").replace("👨🏻‍🚀","").replace("👨🏾‍🚀","").replace("👨🏼‍🚀","").replace("👨🏽‍🚀","").replace("🚴‍♂️","").replace("🚴🏿‍♂️","").replace("🚴🏻‍♂️","").replace("🚴🏾‍♂️","").replace("🚴🏼‍♂️","").replace("🚴🏽‍♂️","").replace("⛹️‍♂️","").replace("⛹🏿‍♂️","").replace("⛹🏻‍♂️","").replace("⛹🏾‍♂️","").replace("⛹🏼‍♂️","").replace("⛹🏽‍♂️","").replace("🙇‍♂️","").replace("🙇🏿‍♂️","").replace("🙇🏻‍♂️","").replace("🙇🏾‍♂️","").replace("🙇🏼‍♂️","").replace("🙇🏽‍♂️","").replace("🤸‍♂️","").replace("🤸🏿‍♂️","").replace("🤸🏻‍♂️","").replace("🤸🏾‍♂️","").replace("🤸🏼‍♂️","").replace("🤸🏽‍♂️","").replace("🧗‍♂️","").replace("🧗🏿‍♂️","").replace("🧗🏻‍♂️","").replace("🧗🏾‍♂️","").replace("🧗🏼‍♂️","").replace("🧗🏽‍♂️","").replace("👷‍♂️","").replace("👷🏿‍♂️","").replace("👷🏻‍♂️","").replace("👷🏾‍♂️","").replace("👷🏼‍♂️","").replace("👷🏽‍♂️","").replace("👨‍🍳","").replace("👨🏿‍🍳","").replace("👨🏻‍🍳","").replace("👨🏾‍🍳","").replace("👨🏼‍🍳","").replace("👨🏽‍🍳","").replace("🕺","").replace("🕺🏿","").replace("🕺🏻","").replace("🕺🏾","").replace("🕺🏼","").replace("🕺🏽","").replace("👨🏿","").replace("🕵️‍♂️","").replace("🕵🏿‍♂️","").replace("🕵🏻‍♂️","").replace("🕵🏾‍♂️","").replace("🕵🏼‍♂️","").replace("🕵🏽‍♂️","").replace("🧝‍♂️","").replace("🧝🏿‍♂️","").replace("🧝🏻‍♂️","").replace("🧝🏾‍♂️","").replace("🧝🏼‍♂️","").replace("🧝🏽‍♂️","").replace("🤦‍♂️","").replace("🤦🏿‍♂️","").replace("🤦🏻‍♂️","").replace("🤦🏾‍♂️","").replace("🤦🏼‍♂️","").replace("🤦🏽‍♂️","").replace("👨‍🏭","").replace("👨🏿‍🏭","").replace("👨🏻‍🏭","").replace("👨🏾‍🏭","").replace("👨🏼‍🏭","").replace("👨🏽‍🏭","").replace("🧚‍♂️","").replace("🧚🏿‍♂️","").replace("🧚🏻‍♂️","").replace("🧚🏾‍♂️","").replace("🧚🏼‍♂️","").replace("🧚🏽‍♂️","").replace("👨‍🌾","").replace("👨🏿‍🌾","").replace("👨🏻‍🌾","").replace("👨🏾‍🌾","").replace("👨🏼‍🌾","").replace("👨🏽‍🌾","").replace("👨‍🚒","").replace("👨🏿‍🚒","").replace("👨🏻‍🚒","").replace("👨🏾‍🚒","").replace("👨🏼‍🚒","").replace("👨🏽‍🚒","").replace("🙍‍♂️","").replace("🙍🏿‍♂️","").replace("🙍🏻‍♂️","").replace("🙍🏾‍♂️","").replace("🙍🏼‍♂️","").replace("🙍🏽‍♂️","").replace("🧞‍♂️","").replace("🙅‍♂️","").replace("🙅🏿‍♂️","").replace("🙅🏻‍♂️","").replace("🙅🏾‍♂️","").replace("🙅🏼‍♂️","").replace("🙅🏽‍♂️","").replace("🙆‍♂️","").replace("🙆🏿‍♂️","").replace("🙆🏻‍♂️","").replace("🙆🏾‍♂️","").replace("🙆🏼‍♂️","").replace("🙆🏽‍♂️","").replace("💇‍♂️","").replace("💇🏿‍♂️","").replace("💇🏻‍♂️","").replace("💇🏾‍♂️","").replace("💇🏼‍♂️","").replace("💇🏽‍♂️","").replace("💆‍♂️","").replace("💆🏿‍♂️","").replace("💆🏻‍♂️","").replace("💆🏾‍♂️","").replace("💆🏼‍♂️","").replace("💆🏽‍♂️","").replace("🏌️‍♂️","").replace("🏌🏿‍♂️","").replace("🏌🏻‍♂️","").replace("🏌🏾‍♂️","").replace("🏌🏼‍♂️","").replace("🏌🏽‍♂️","").replace("💂‍♂️","").replace("💂🏿‍♂️","").replace("💂🏻‍♂️","").replace("💂🏾‍♂️","").replace("💂🏼‍♂️","").replace("💂🏽‍♂️","").replace("👨‍⚕️","").replace("👨🏿‍⚕️","").replace("👨🏻‍⚕️","").replace("👨🏾‍⚕️","").replace("👨🏼‍⚕️","").replace("👨🏽‍⚕️","").replace("🧘‍♂️","").replace("🧘🏿‍♂️","").replace("🧘🏻‍♂️","").replace("🧘🏾‍♂️","").replace("🧘🏼‍♂️","").replace("🧘🏽‍♂️","").replace("🧖‍♂️","").replace("🧖🏿‍♂️","").replace("🧖🏻‍♂️","").replace("🧖🏾‍♂️","").replace("🧖🏼‍♂️","").replace("🧖🏽‍♂️","").replace("🕴","").replace("🕴🏿","").replace("🕴🏻","").replace("🕴🏾","").replace("🕴🏼","").replace("🕴🏽","").replace("🤵","").replace("🤵🏿","").replace("🤵🏻","").replace("🤵🏾","").replace("🤵🏼","").replace("🤵🏽","").replace("👨‍⚖️","").replace("👨🏿‍⚖️","").replace("👨🏻‍⚖️","").replace("👨🏾‍⚖️","").replace("👨🏼‍⚖️","").replace("👨🏽‍⚖️","").replace("🤹‍♂️","").replace("🤹🏿‍♂️","").replace("🤹🏻‍♂️","").replace("🤹🏾‍♂️","").replace("🤹🏼‍♂️","").replace("🤹🏽‍♂️","").replace("🏋️‍♂️","").replace("🏋🏿‍♂️","").replace("🏋🏻‍♂️","").replace("🏋🏾‍♂️","").replace("🏋🏼‍♂️","").replace("🏋🏽‍♂️","").replace("👨🏻","").replace("🧙‍♂️","").replace("🧙🏿‍♂️","").replace("🧙🏻‍♂️","").replace("🧙🏾‍♂️","").replace("🧙🏼‍♂️","").replace("🧙🏽‍♂️","").replace("👨‍🔧","").replace("👨🏿‍🔧","").replace("👨🏻‍🔧","").replace("👨🏾‍🔧","").replace("👨🏼‍🔧","").replace("👨🏽‍🔧","").replace("👨🏾","").replace("👨🏼","").replace("👨🏽","").replace("🚵‍♂️","").replace("🚵🏿‍♂️","").replace("🚵🏻‍♂️","").replace("🚵🏾‍♂️","").replace("🚵🏼‍♂️","").replace("🚵🏽‍♂️","").replace("👨‍💼","").replace("👨🏿‍💼","").replace("👨🏻‍💼","").replace("👨🏾‍💼","").replace("👨🏼‍💼","").replace("👨🏽‍💼","").replace("👨‍✈️","").replace("👨🏿‍✈️","").replace("👨🏻‍✈️","").replace("👨🏾‍✈️","").replace("👨🏼‍✈️","").replace("👨🏽‍✈️","").replace("🤾‍♂️","").replace("🤾🏿‍♂️","").replace("🤾🏻‍♂️","").replace("🤾🏾‍♂️","").replace("🤾🏼‍♂️","").replace("🤾🏽‍♂️","").replace("🤽‍♂️","").replace("🤽🏿‍♂️","").replace("🤽🏻‍♂️","").replace("🤽🏾‍♂️","").replace("🤽🏼‍♂️","").replace("🤽🏽‍♂️","").replace("👮‍♂️","").replace("👮🏿‍♂️","").replace("👮🏻‍♂️","").replace("👮🏾‍♂️","").replace("👮🏼‍♂️","").replace("👮🏽‍♂️","").replace("🙎‍♂️","").replace("🙎🏿‍♂️","").replace("🙎🏻‍♂️","").replace("🙎🏾‍♂️","").replace("🙎🏼‍♂️","").replace("🙎🏽‍♂️","").replace("🙋‍♂️","").replace("🙋🏿‍♂️","").replace("🙋🏻‍♂️","").replace("🙋🏾‍♂️","").replace("🙋🏼‍♂️","").replace("🙋🏽‍♂️","").replace("🚣‍♂️","").replace("🚣🏿‍♂️","").replace("🚣🏻‍♂️","").replace("🚣🏾‍♂️","").replace("🚣🏼‍♂️","").replace("🚣🏽‍♂️","").replace("🏃‍♂️","").replace("🏃🏿‍♂️","").replace("🏃🏻‍♂️","").replace("🏃🏾‍♂️","").replace("🏃🏼‍♂️","").replace("🏃🏽‍♂️","").replace("👨‍🔬","").replace("👨🏿‍🔬","").replace("👨🏻‍🔬","").replace("👨🏾‍🔬","").replace("👨🏼‍🔬","").replace("👨🏽‍🔬","").replace("🤷‍♂️","").replace("🤷🏿‍♂️","").replace("🤷🏻‍♂️","").replace("🤷🏾‍♂️","").replace("🤷🏼‍♂️","").replace("🤷🏽‍♂️","").replace("👨‍🎤","").replace("👨🏿‍🎤","").replace("👨🏻‍🎤","").replace("👨🏾‍🎤","").replace("👨🏼‍🎤","").replace("👨🏽‍🎤","").replace("👨‍🎓","").replace("👨🏿‍🎓","").replace("👨🏻‍🎓","").replace("👨🏾‍🎓","").replace("👨🏼‍🎓","").replace("👨🏽‍🎓","").replace("🏄‍♂️","").replace("🏄🏿‍♂️","").replace("🏄🏻‍♂️","").replace("🏄🏾‍♂️","").replace("🏄🏼‍♂️","").replace("🏄🏽‍♂️","").replace("🏊‍♂️","").replace("🏊🏿‍♂️","").replace("🏊🏻‍♂️","").replace("🏊🏾‍♂️","").replace("🏊🏼‍♂️","").replace("🏊🏽‍♂️","").replace("👨‍🏫","").replace("👨🏿‍🏫","").replace("👨🏻‍🏫","").replace("👨🏾‍🏫","").replace("👨🏼‍🏫","").replace("👨🏽‍🏫","").replace("👨‍💻","").replace("👨🏿‍💻","").replace("👨🏻‍💻","").replace("👨🏾‍💻","").replace("👨🏼‍💻","").replace("👨🏽‍💻","").replace("💁‍♂️","").replace("💁🏿‍♂️","").replace("💁🏻‍♂️","").replace("💁🏾‍♂️","").replace("💁🏼‍♂️","").replace("💁🏽‍♂️","").replace("🧛‍♂️","").replace("🧛🏿‍♂️","").replace("🧛🏻‍♂️","").replace("🧛🏾‍♂️","").replace("🧛🏼‍♂️","").replace("🧛🏽‍♂️","").replace("🚶‍♂️","").replace("🚶🏿‍♂️","").replace("🚶🏻‍♂️","").replace("🚶🏾‍♂️","").replace("🚶🏼‍♂️","").replace("🚶🏽‍♂️","").replace("👳‍♂️","").replace("👳🏿‍♂️","").replace("👳🏻‍♂️","").replace("👳🏾‍♂️","").replace("👳🏼‍♂️","").replace("👳🏽‍♂️","").replace("👲","").replace("👲🏿","").replace("👲🏻","").replace("👲🏾","").replace("👲🏼","").replace("👲🏽","").replace("🧟‍♂️","").replace("🕰","").replace("👞","").replace("🗾","").replace("🍁","").replace("🥋","").replace("🍖","").replace("⚕","").replace("📣","").replace("🍈","").replace("📝","").replace("👯‍♂️","").replace("🤼‍♂️","").replace("🕎","").replace("🚹","").replace("🧜‍♀️","").replace("🧜🏿‍♀️","").replace("🧜🏻‍♀️","").replace("🧜🏾‍♀️","").replace("🧜🏼‍♀️","").replace("🧜🏽‍♀️","").replace("🧜‍♂️","").replace("🧜🏿‍♂️","").replace("🧜🏻‍♂️","").replace("🧜🏾‍♂️","").replace("🧜🏼‍♂️","").replace("🧜🏽‍♂️","").replace("🧜","").replace("🧜🏿","").replace("🧜🏻","").replace("🧜🏾","").replace("🧜🏼","").replace("🧜🏽","").replace("🚇","").replace("🎤","").replace("🔬","").replace("🖕","").replace("🖕🏿","").replace("🖕🏻","").replace("🖕🏾","").replace("🖕🏼","").replace("🖕🏽","").replace("🎖","").replace("🌌","").replace("🚐","").replace("🗿","").replace("📱","").replace("📴","").replace("📲","").replace("🤑","").replace("💰","").replace("💸","").replace("🐒","").replace("🐵","").replace("🚝","").replace("🎑","").replace("🕌","").replace("🛥","").replace("🛵","").replace("🏍","").replace("🛣","").replace("🗻","").replace("⛰","").replace("🚠","").replace("🚞","").replace("🐭","").replace("🐭","").replace("👄","").replace("🎥","").replace("🍄","").replace("🎹","").replace("🎵","").replace("🎶","").replace("🎼","").replace("🔇","").replace("💅","").replace("💅🏿","").replace("💅🏻","").replace("💅🏾","").replace("💅🏼","").replace("💅🏽","").replace("📛","").replace("🏞","").replace("🤢","").replace("👔","").replace("🤓","").replace("😐","").replace("🌑","").replace("🌚","").replace("📰","").replace("⏭","").replace("🌃","").replace("🕤","").replace("🕘","").replace("🚳","").replace("⛔","").replace("🚯","").replace("📵","").replace("🔞","").replace("🚷","").replace("🚭","").replace("🚱","").replace("👃","").replace("👃🏿","").replace("👃🏻","").replace("👃🏾","").replace("👃🏼","").replace("👃🏽","").replace("📓","").replace("📔","").replace("🔩","").replace("🐙","").replace("🍢","").replace("🏢","").replace("👹","").replace("🛢","").replace("🗝","").replace("👴","").replace("👴🏿","").replace("👴🏻","").replace("👴🏾","").replace("👴🏼","").replace("👴🏽","").replace("👵","").replace("👵🏿","").replace("👵🏻","").replace("👵🏾","").replace("👵🏼","").replace("👵🏽","").replace("🧓","").replace("🧓🏿","").replace("🧓🏻","").replace("🧓🏾","").replace("🧓🏼","").replace("🧓🏽","").replace("🕉","").replace("🚘","").replace("🚍","").replace("👊","").replace("👊🏿","").replace("👊🏻","").replace("👊🏾","").replace("👊🏼","").replace("👊🏽","").replace("🚔","").replace("🚖","").replace("🕜","").replace("🕐","").replace("📖","").replace("📂","").replace("👐","").replace("👐🏿","").replace("👐🏻","").replace("👐🏾","").replace("👐🏼","").replace("👐🏽","").replace("📭","").replace("📬","").replace("💿","").replace("📙","").replace("🧡","").replace("☦","").replace("📤","").replace("🦉","").replace("🐂","").replace("📦","").replace("📄","").replace("📃","").replace("📟","").replace("🖌","").replace("🌴","").replace("🤲","").replace("🤲🏿","").replace("🤲🏻","").replace("🤲🏾","").replace("🤲🏼","").replace("🤲🏽","").replace("🥞","").replace("🐼","").replace("📎","").replace("〽","").replace("🎉","").replace("🛳","").replace("🛂","").replace("⏸","").replace("🐾","").replace("☮","").replace("🍑","").replace("🥜","").replace("🍐","").replace("🖊","").replace("📝","").replace("🐧","").replace("😔","").replace("👯","").replace("🤼","").replace("🎭","").replace("😣","").replace("🚴","").replace("🚴🏿","").replace("🚴🏻","").replace("🚴🏾","").replace("🚴🏼","").replace("🚴🏽","").replace("⛹","").replace("⛹🏿","").replace("⛹🏻","").replace("⛹🏾","").replace("⛹🏼","").replace("⛹🏽","").replace("🙇","").replace("🙇🏿","").replace("🙇🏻","").replace("🙇🏾","").replace("🙇🏼","").replace("🙇🏽","").replace("🤸","").replace("🤸🏿","").replace("🤸🏻","").replace("🤸🏾","").replace("🤸🏼","").replace("🤸🏽","").replace("🧗","").replace("🧗🏿","").replace("🧗🏻","").replace("🧗🏾","").replace("🧗🏼","").replace("🧗🏽","").replace("🤦","").replace("🤦🏿","").replace("🤦🏻","").replace("🤦🏾","").replace("🤦🏼","").replace("🤦🏽","").replace("🤺","").replace("🙍","").replace("🙍🏿","").replace("🙍🏻","").replace("🙍🏾","").replace("🙍🏼","").replace("🙍🏽","").replace("🙅","").replace("🙅🏿","").replace("🙅🏻","").replace("🙅🏾","").replace("🙅🏼","").replace("🙅🏽","").replace("🙆","").replace("🙆🏿","").replace("🙆🏻","").replace("🙆🏾","").replace("🙆🏼","").replace("🙆🏽","").replace("💇","").replace("💇🏿","").replace("💇🏻","").replace("💇🏾","").replace("💇🏼","").replace("💇🏽","").replace("💆","").replace("💆🏿","").replace("💆🏻","").replace("💆🏾","").replace("💆🏼","").replace("💆🏽","").replace("🏌","").replace("🏌🏿","").replace("🏌🏻","").replace("🏌🏾","").replace("🏌🏼","").replace("🏌🏽","").replace("🛌","").replace("🛌🏿","").replace("🛌🏻","").replace("🛌🏾","").replace("🛌🏼","").replace("🛌🏽","").replace("🧘","").replace("🧘🏿","").replace("🧘🏻","").replace("🧘🏾","").replace("🧘🏼","").replace("🧘🏽","").replace("🧖","").replace("🧖🏿","").replace("🧖🏻","").replace("🧖🏾","").replace("🧖🏼","").replace("🧖🏽","").replace("🤹","").replace("🤹🏿","").replace("🤹🏻","").replace("🤹🏾","").replace("🤹🏼","").replace("🤹🏽","").replace("🏋","").replace("🏋🏿","").replace("🏋🏻","").replace("🏋🏾","").replace("🏋🏼","").replace("🏋🏽","").replace("🚵","").replace("🚵🏿","").replace("🚵🏻","").replace("🚵🏾","").replace("🚵🏼","").replace("🚵🏽","").replace("🤾","").replace("🤾🏿","").replace("🤾🏻","").replace("🤾🏾","").replace("🤾🏼","").replace("🤾🏽","").replace("🤽","").replace("🤽🏿","").replace("🤽🏻","").replace("🤽🏾","").replace("🤽🏼","").replace("🤽🏽","").replace("🙎","").replace("🙎🏿","").replace("🙎🏻","").replace("🙎🏾","").replace("🙎🏼","").replace("🙎🏽","").replace("🙋","").replace("🙋🏿","").replace("🙋🏻","").replace("🙋🏾","").replace("🙋🏼","").replace("🙋🏽","").replace("🚣","").replace("🚣🏿","").replace("🚣🏻","").replace("🚣🏾","").replace("🚣🏼","").replace("🚣🏽","").replace("🏃","").replace("🏃🏿","").replace("🏃🏻","").replace("🏃🏾","").replace("🏃🏼","").replace("🏃🏽","").replace("🤷","").replace("🤷🏿","").replace("🤷🏻","").replace("🤷🏾","").replace("🤷🏼","").replace("🤷🏽","").replace("🏄","").replace("🏄🏿","").replace("🏄🏻","").replace("🏄🏾","").replace("🏄🏼","").replace("🏄🏽","").replace("🏊","").replace("🏊🏿","").replace("🏊🏻","").replace("🏊🏾","").replace("🏊🏼","").replace("🏊🏽","").replace("🛀","").replace("🛀🏿","").replace("🛀🏻","").replace("🛀🏾","").replace("🛀🏼","").replace("🛀🏽","").replace("💁","").replace("💁🏿","").replace("💁🏻","").replace("💁🏾","").replace("💁🏼","").replace("💁🏽","").replace("🚶","").replace("🚶🏿","").replace("🚶🏻","").replace("🚶🏾","").replace("🚶🏼","").replace("🚶🏽","").replace("👳","").replace("👳🏿","").replace("👳🏻","").replace("👳🏾","").replace("👳🏼","").replace("👳🏽","").replace("⛏","").replace("🥧","").replace("🐷","").replace("🐷","").replace("🐽","").replace("💩","").replace("💊","").replace("🎍","").replace("🍍","").replace("🏓","").replace("🔫","").replace("🍕","").replace("🛐","").replace("▶","").replace("⏯","").replace("🚓","").replace("🚨","").replace("👮","").replace("👮🏿","").replace("👮🏻","").replace("👮🏾","").replace("👮🏼","").replace("👮🏽","").replace("🐩","").replace("🎱","").replace("🍿","").replace("🏣","").replace("📯","").replace("📮","").replace("🍲","").replace("🚰","").replace("🥔","").replace("🍗","").replace("💷","").replace("😾","").replace("😡","").replace("📿","").replace("🤰","").replace("🤰🏿","").replace("🤰🏻","").replace("🤰🏾","").replace("🤰🏼","").replace("🤰🏽","").replace("🥨","").replace("🤴","").replace("🤴🏿","").replace("🤴🏻","").replace("🤴🏾","").replace("🤴🏼","").replace("🤴🏽","").replace("👸","").replace("👸🏿","").replace("👸🏻","").replace("👸🏾","").replace("👸🏼","").replace("👸🏽","").replace("🖨","").replace("🚫","").replace("💜","").replace("👛","").replace("📌","").replace("❓","").replace("🐰","").replace("🐰","").replace("🏎","").replace("📻","").replace("🔘","").replace("☢","").replace("🚃","").replace("🛤","").replace("🌈","").replace("🏳️‍🌈","").replace("🤚","").replace("🤚🏿","").replace("🤚🏻","").replace("🤚🏾","").replace("🤚🏼","").replace("🤚🏽","").replace("✊","").replace("✊🏿","").replace("✊🏻","").replace("✊🏾","").replace("✊🏼","").replace("✊🏽","").replace("✋","").replace("✋🏿","").replace("✋🏻","").replace("✋🏾","").replace("✋🏼","").replace("✋🏽","").replace("🙌","").replace("🙌🏿","").replace("🙌🏻","").replace("🙌🏾","").replace("🙌🏼","").replace("🙌🏽","").replace("🐏","").replace("🐀","").replace("⏺","").replace("♻","").replace("🍎","").replace("🔴","").replace("❤","").replace("🏮","").replace("🔻","").replace("🔺","").replace("®","").replace("😌","").replace("🎗","").replace("🔁","").replace("🔂","").replace("⛑","").replace("🚻","").replace("◀","").replace("💞","").replace("🦏","").replace("🎀","").replace("🍙","").replace("🍘","").replace("🤜","").replace("🤜🏿","").replace("🤜🏻","").replace("🤜🏾","").replace("🤜🏼","").replace("🤜🏽","").replace("🗯","").replace("➡","").replace("⤵","").replace("↩","").replace("⤴","").replace("💍","").replace("🍠","").replace("🤖","").replace("🚀","").replace("🗞","").replace("🎢","").replace("🤣","").replace("🐓","").replace("🌹","").replace("🏵","").replace("📍","").replace("🏉","").replace("🎽","").replace("👟","").replace("😥","").replace("⛵","").replace("🍶","").replace("🥪","").replace("📡","").replace("📡","").replace("🦕","").replace("🎷","").replace("🧣","").replace("🏫","").replace("🎒","").replace("✂","").replace("🦂","").replace("📜","").replace("💺","").replace("🙈","").replace("🌱","").replace("🤳","").replace("🤳🏿","").replace("🤳🏻","").replace("🤳🏾","").replace("🤳🏼","").replace("🤳🏽","").replace("🕢","").replace("🕖","").replace("🥘","").replace("☘","").replace("🦈","").replace("🍧","").replace("🌾","").replace("🛡","").replace("⛩","").replace("🚢","").replace("🌠","").replace("🛍","").replace("🛒","").replace("🍰","").replace("🚿","").replace("🦐","").replace("🔀","").replace("🤫","").replace("🤘","").replace("🤘🏿","").replace("🤘🏻","").replace("🤘🏾","").replace("🤘🏼","").replace("🤘🏽","").replace("🕡","").replace("🕕","").replace("⛷","").replace("🎿","").replace("💀","").replace("☠","").replace("🛷","").replace("😴","").replace("😪","").replace("🙁","").replace("🙂","").replace("🎰","").replace("🛩","").replace("🔹","").replace("🔸","").replace("😻","").replace("☺","").replace("😇","").replace("😍","").replace("😈","").replace("😊","").replace("😎","").replace("😏","").replace("🐌","").replace("🐍","").replace("🤧","").replace("🏔","").replace("🏂","").replace("🏂🏿","").replace("🏂🏻","").replace("🏂🏾","").replace("🏂🏼","").replace("🏂🏽","").replace("❄","").replace("☃","").replace("⛄","").replace("⚽","").replace("🧦","").replace("🍦","").replace("♠","").replace("🍝","").replace("❇","").replace("🎇","").replace("✨","").replace("💖","").replace("🙊","").replace("🔊","").replace("🔈","").replace("🔉","").replace("🗣","").replace("💬","").replace("🚤","").replace("🕷","").replace("🕸","").replace("🗓","").replace("🗒","").replace("🐚","").replace("🥄","").replace("🚙","").replace("🏅","").replace("🐳","").replace("🦑","").replace("😝","").replace("🏟","").replace("🤩","").replace("☪","").replace("✡","").replace("🚉","").replace("🍜","").replace("⏹","").replace("🛑","").replace("⏱","").replace("📏","").replace("🍓","").replace("🎙","").replace("🥙","").replace("☀","").replace("⛅","").replace("🌥","").replace("🌦","").replace("🌤","").replace("🌞","").replace("🌻","").replace("😎","").replace("🌅","").replace("🌄","").replace("🌇","").replace("🍣","").replace("🚟","").replace("💦","").replace("🕍","").replace("💉","").replace("👕","").replace("🌮","").replace("🥡","").replace("🎋","").replace("🍊","").replace("🚕","").replace("🍵","").replace("📆","").replace("☎","").replace("📞","").replace("🔭","").replace("📺","").replace("🕥","").replace("🕙","").replace("🎾","").replace("⛺","").replace("🌡","").replace("🤔","").replace("💭","").replace("🕞","").replace("🕒","").replace("👎","").replace("👎🏿","").replace("👎🏻","").replace("👎🏾","").replace("👎🏼","").replace("👎🏽","").replace("👍","").replace("👍🏿","").replace("👍🏻","").replace("👍🏾","").replace("👍🏼","").replace("👍🏽","").replace("🎫","").replace("🐯","").replace("🐯","").replace("⏲","").replace("😫","").replace("🚽","").replace("🍅","").replace("👅","").replace("🎩","").replace("🌪","").replace("🖲","").replace("🚜","").replace("™","").replace("🚋","").replace("🚊","").replace("🚋","").replace("🚩","").replace("📐","").replace("🔱","").replace("🚎","").replace("🏆","").replace("🍹","").replace("🐠","").replace("🎺","").replace("🌷","").replace("🥃","").replace("🦃","").replace("🐢","").replace("🕧","").replace("🕛","").replace("🐫","").replace("🕝","").replace("💕","").replace("👬","").replace("🕑","").replace("👭","").replace("☂","").replace("⛱","").replace("☔","").replace("😒","").replace("🦄","").replace("🔓","").replace("↕","").replace("↖","").replace("↗","").replace("⬆","").replace("🙃","").replace("🔼","").replace("🧛","").replace("🧛🏿","").replace("🧛🏻","").replace("🧛🏾","").replace("🧛🏼","").replace("🧛🏽","").replace("🚦","").replace("📳","").replace("✌","").replace("✌🏿","").replace("✌🏻","").replace("✌🏾","").replace("✌🏼","").replace("✌🏽","").replace("📹","").replace("🎮","").replace("📼","").replace("🎻","").replace("🌋","").replace("🏐","").replace("🖖","").replace("🖖🏿","").replace("🖖🏻","").replace("🖖🏾","").replace("🖖🏼","").replace("🖖🏽","").replace("🌘","").replace("🌖","").replace("⚠","").replace("🗑","").replace("⌚","").replace("🐃","").replace("🚾","").replace("🌊","").replace("🍉","").replace("👋","").replace("👋🏿","").replace("👋🏻","").replace("👋🏾","").replace("👋🏼","").replace("👋🏽","").replace("〰","").replace("🌒","").replace("🌔","").replace("🙀","").replace("😩","").replace("💒","").replace("🐳","").replace("☸","").replace("♿","").replace("⚪","").replace("❕","").replace("🏳","").replace("💮","").replace("✅","").replace("⬜","").replace("◽","").replace("◻","").replace("⭐","").replace("❔","").replace("▫","").replace("🔳","").replace("🥀","").replace("🎐","").replace("🌬","").replace("🍷","").replace("😉","").replace("😜","").replace("🐺","").replace("👩","").replace("👩‍🎨","").replace("👩🏿‍🎨","").replace("👩🏻‍🎨","").replace("👩🏾‍🎨","").replace("👩🏼‍🎨","").replace("👩🏽‍🎨","").replace("👩‍🚀","").replace("👩🏿‍🚀","").replace("👩🏻‍🚀","").replace("👩🏾‍🚀","").replace("👩🏼‍🚀","").replace("👩🏽‍🚀","").replace("🚴‍♀️","").replace("🚴🏿‍♀️","").replace("🚴🏻‍♀️","").replace("🚴🏾‍♀️","").replace("🚴🏼‍♀️","").replace("🚴🏽‍♀️","").replace("⛹️‍♀️","").replace("⛹🏿‍♀️","").replace("⛹🏻‍♀️","").replace("⛹🏾‍♀️","").replace("⛹🏼‍♀️","").replace("⛹🏽‍♀️","").replace("🙇‍♀️","").replace("🙇🏿‍♀️","").replace("🙇🏻‍♀️","").replace("🙇🏾‍♀️","").replace("🙇🏼‍♀️","").replace("🙇🏽‍♀️","").replace("🤸‍♀️","").replace("🤸🏿‍♀️","").replace("🤸🏻‍♀️","").replace("🤸🏾‍♀️","").replace("🤸🏼‍♀️","").replace("🤸🏽‍♀️","").replace("🧗‍♀️","").replace("🧗🏿‍♀️","").replace("🧗🏻‍♀️","").replace("🧗🏾‍♀️","").replace("🧗🏼‍♀️","").replace("🧗🏽‍♀️","").replace("👷‍♀️","").replace("👷🏿‍♀️","").replace("👷🏻‍♀️","").replace("👷🏾‍♀️","").replace("👷🏼‍♀️","").replace("👷🏽‍♀️","").replace("👩‍🍳","").replace("👩🏿‍🍳","").replace("👩🏻‍🍳","").replace("👩🏾‍🍳","").replace("👩🏼‍🍳","").replace("👩🏽‍🍳","").replace("💃","").replace("💃🏿","").replace("💃🏻","").replace("💃🏾","").replace("💃🏼","").replace("💃🏽","").replace("👩🏿","").replace("🕵️‍♀️","").replace("🕵🏿‍♀️","").replace("🕵🏻‍♀️","").replace("🕵🏾‍♀️","").replace("🕵🏼‍♀️","").replace("🕵🏽‍♀️","").replace("🧝‍♀️","").replace("🧝🏿‍♀️","").replace("🧝🏻‍♀️","").replace("🧝🏾‍♀️","").replace("🧝🏼‍♀️","").replace("🧝🏽‍♀️","").replace("🤦‍♀️","").replace("🤦🏿‍♀️","").replace("🤦🏻‍♀️","").replace("🤦🏾‍♀️","").replace("🤦🏼‍♀️","").replace("🤦🏽‍♀️","").replace("👩‍🏭","").replace("👩🏿‍🏭","").replace("👩🏻‍🏭","").replace("👩🏾‍🏭","").replace("👩🏼‍🏭","").replace("👩🏽‍🏭","").replace("🧚‍♀️","").replace("🧚🏿‍♀️","").replace("🧚🏻‍♀️","").replace("🧚🏾‍♀️","").replace("🧚🏼‍♀️","").replace("🧚🏽‍♀️","").replace("👩‍🌾","").replace("👩🏿‍🌾","").replace("👩🏻‍🌾","").replace("👩🏾‍🌾","").replace("👩🏼‍🌾","").replace("👩🏽‍🌾","").replace("👩‍🚒","").replace("👩🏿‍🚒","").replace("👩🏻‍🚒","").replace("👩🏾‍🚒","").replace("👩🏼‍🚒","").replace("👩🏽‍🚒","").replace("🙍‍♀️","").replace("🙍🏿‍♀️","").replace("🙍🏻‍♀️","").replace("🙍🏾‍♀️","").replace("🙍🏼‍♀️","").replace("🙍🏽‍♀️","").replace("🧞‍♀️","").replace("🙅‍♀️","").replace("🙅🏿‍♀️","").replace("🙅🏻‍♀️","").replace("🙅🏾‍♀️","").replace("🙅🏼‍♀️","").replace("🙅🏽‍♀️","").replace("🙆‍♀️","").replace("🙆🏿‍♀️","").replace("🙆🏻‍♀️","").replace("🙆🏾‍♀️","").replace("🙆🏼‍♀️","").replace("🙆🏽‍♀️","").replace("💇‍♀️","").replace("💇🏿‍♀️","").replace("💇🏻‍♀️","").replace("💇🏾‍♀️","").replace("💇🏼‍♀️","").replace("💇🏽‍♀️","").replace("💆‍♀️","").replace("💆🏿‍♀️","").replace("💆🏻‍♀️","").replace("💆🏾‍♀️","").replace("💆🏼‍♀️","").replace("💆🏽‍♀️","").replace("🏌️‍♀️","").replace("🏌🏿‍♀️","").replace("🏌🏻‍♀️","").replace("🏌🏾‍♀️","").replace("🏌🏼‍♀️","").replace("🏌🏽‍♀️","").replace("💂‍♀️","").replace("💂🏿‍♀️","").replace("💂🏻‍♀️","").replace("💂🏾‍♀️","").replace("💂🏼‍♀️","").replace("💂🏽‍♀️","").replace("👩‍⚕️","").replace("👩🏿‍⚕️","").replace("👩🏻‍⚕️","").replace("👩🏾‍⚕️","").replace("👩🏼‍⚕️","").replace("👩🏽‍⚕️","").replace("🧘‍♀️","").replace("🧘🏿‍♀️","").replace("🧘🏻‍♀️","").replace("🧘🏾‍♀️","").replace("🧘🏼‍♀️","").replace("🧘🏽‍♀️","").replace("🧖‍♀️","").replace("🧖🏿‍♀️","").replace("🧖🏻‍♀️","").replace("🧖🏾‍♀️","").replace("🧖🏼‍♀️","").replace("🧖🏽‍♀️","").replace("👩‍⚖️","").replace("👩🏿‍⚖️","").replace("👩🏻‍⚖️","").replace("👩🏾‍⚖️","").replace("👩🏼‍⚖️","").replace("👩🏽‍⚖️","").replace("🤹‍♀️","").replace("🤹🏿‍♀️","").replace("🤹🏻‍♀️","").replace("🤹🏾‍♀️","").replace("🤹🏼‍♀️","").replace("🤹🏽‍♀️","").replace("🏋️‍♀️","").replace("🏋🏿‍♀️","").replace("🏋🏻‍♀️","").replace("🏋🏾‍♀️","").replace("🏋🏼‍♀️","").replace("🏋🏽‍♀️","").replace("👩🏻","").replace("🧙‍♀️","").replace("🧙🏿‍♀️","").replace("🧙🏻‍♀️","").replace("🧙🏾‍♀️","").replace("🧙🏼‍♀️","").replace("🧙🏽‍♀️","").replace("👩‍🔧","").replace("👩🏿‍🔧","").replace("👩🏻‍🔧","").replace("👩🏾‍🔧","").replace("👩🏼‍🔧","").replace("👩🏽‍🔧","").replace("👩🏾","").replace("👩🏼","").replace("👩🏽","").replace("🚵‍♀️","").replace("🚵🏿‍♀️","").replace("🚵🏻‍♀️","").replace("🚵🏾‍♀️","").replace("🚵🏼‍♀️","").replace("🚵🏽‍♀️","").replace("👩‍💼","").replace("👩🏿‍💼","").replace("👩🏻‍💼","").replace("👩🏾‍💼","").replace("👩🏼‍💼","").replace("👩🏽‍💼","").replace("👩‍✈️","").replace("👩🏿‍✈️","").replace("👩🏻‍✈️","").replace("👩🏾‍✈️","").replace("👩🏼‍✈️","").replace("👩🏽‍✈️","").replace("🤾‍♀️","").replace("🤾🏿‍♀️","").replace("🤾🏻‍♀️","").replace("🤾🏾‍♀️","").replace("🤾🏼‍♀️","").replace("🤾🏽‍♀️","").replace("🤽‍♀️","").replace("🤽🏿‍♀️","").replace("🤽🏻‍♀️","").replace("🤽🏾‍♀️","").replace("🤽🏼‍♀️","").replace("🤽🏽‍♀️","").replace("👮‍♀️","").replace("👮🏿‍♀️","").replace("👮🏻‍♀️","").replace("👮🏾‍♀️","").replace("👮🏼‍♀️","").replace("👮🏽‍♀️","").replace("🙎‍♀️","").replace("🙎🏿‍♀️","").replace("🙎🏻‍♀️","").replace("🙎🏾‍♀️","").replace("🙎🏼‍♀️","").replace("🙎🏽‍♀️","").replace("🙋‍♀️","").replace("🙋🏿‍♀️","").replace("🙋🏻‍♀️","").replace("🙋🏾‍♀️","").replace("🙋🏼‍♀️","").replace("🙋🏽‍♀️","").replace("🚣‍♀️","").replace("🚣🏿‍♀️","").replace("🚣🏻‍♀️","").replace("🚣🏾‍♀️","").replace("🚣🏼‍♀️","").replace("🚣🏽‍♀️","").replace("🏃‍♀️","").replace("🏃🏿‍♀️","").replace("🏃🏻‍♀️","").replace("🏃🏾‍♀️","").replace("🏃🏼‍♀️","").replace("🏃🏽‍♀️","").replace("👩‍🔬","").replace("👩🏿‍🔬","").replace("👩🏻‍🔬","").replace("👩🏾‍🔬","").replace("👩🏼‍🔬","").replace("👩🏽‍🔬","").replace("🤷‍♀️","").replace("🤷🏿‍♀️","").replace("🤷🏻‍♀️","").replace("🤷🏾‍♀️","").replace("🤷🏼‍♀️","").replace("🤷🏽‍♀️","").replace("👩‍🎤","").replace("👩🏿‍🎤","").replace("👩🏻‍🎤","").replace("👩🏾‍🎤","").replace("👩🏼‍🎤","").replace("👩🏽‍🎤","").replace("👩‍🎓","").replace("👩🏿‍🎓","").replace("👩🏻‍🎓","").replace("👩🏾‍🎓","").replace("👩🏼‍🎓","").replace("👩🏽‍🎓","").replace("🏄‍♀️","").replace("🏄🏿‍♀️","").replace("🏄🏻‍♀️","").replace("🏄🏾‍♀️","").replace("🏄🏼‍♀️","").replace("🏄🏽‍♀️","").replace("🏊‍♀️","").replace("🏊🏿‍♀️","").replace("🏊🏻‍♀️","").replace("🏊🏾‍♀️","").replace("🏊🏼‍♀️","").replace("🏊🏽‍♀️","").replace("👩‍🏫","").replace("👩🏿‍🏫","").replace("👩🏻‍🏫","").replace("👩🏾‍🏫","").replace("👩🏼‍🏫","").replace("👩🏽‍🏫","").replace("👩‍💻","").replace("👩🏿‍💻","").replace("👩🏻‍💻","").replace("👩🏾‍💻","").replace("👩🏼‍💻","").replace("👩🏽‍💻","").replace("💁‍♀️","").replace("💁🏿‍♀️","").replace("💁🏻‍♀️","").replace("💁🏾‍♀️","").replace("💁🏼‍♀️","").replace("💁🏽‍♀️","").replace("🧛‍♀️","").replace("🧛🏿‍♀️","").replace("🧛🏻‍♀️","").replace("🧛🏾‍♀️","").replace("🧛🏼‍♀️","").replace("🧛🏽‍♀️","").replace("🚶‍♀️","").replace("🚶🏿‍♀️","").replace("🚶🏻‍♀️","").replace("🚶🏾‍♀️","").replace("🚶🏼‍♀️","").replace("🚶🏽‍♀️","").replace("👳‍♀️","").replace("👳🏿‍♀️","").replace("👳🏻‍♀️","").replace("👳🏾‍♀️","").replace("👳🏼‍♀️","").replace("👳🏽‍♀️","").replace("🧕","").replace("🧕🏿","").replace("🧕🏻","").replace("🧕🏾","").replace("🧕🏼","").replace("🧕🏽","").replace("🧟‍♀️","").replace("👢","").replace("👚","").replace("👒","").replace("👡","").replace("👯‍♀️","").replace("🤼‍♀️","").replace("🚺","").replace("🗺","").replace("😟","").replace("🎁","").replace("🔧","").replace("✍","").replace("✍🏿","").replace("✍🏻","").replace("✍🏾","").replace("✍🏼","").replace("✍🏽","").replace("💛","").replace("💴","").replace("☯","").replace("🤪","").replace("🦓","").replace("🤐","").replace("🧟","").replace("💤","").replace("🇦🇽","")
        #    text1=meta
        #    print text1
        #     text2=split(text1,2000)
        text2=re.split('[!"#$%\()*+,-../:;<=>?@[\\]^_`{|}~]',re.sub('\S+@\S+|@\S+','',text1).strip())
#        text2=text1
        
        text=[]
        text3=text4=text5=regexdf=""
        
        for k in range(len(text2)):
            if len(text2[k])>4000:
                text3=split(text2,4000)
                for l in range(len(text3)):
                    regexdf=""
                    if len(text3[l]) > 2:
                        temptext=""
                        text4=""
                        temptxt=re.sub('[A-Za-z0-9 ]+', ' ', text3[l])
                        text4=re.sub('[^A-Za-z0-9 ]+', ' ', text3[l])
                        if len(temptxt.strip()) !=0:
                            regexdf=' '.join(temptxt.split()).lower()
        #                         if l!=0:    
                            try:
                                username_box.send_keys(".".decode('utf8'))
                                clearbox.click()
            #                         print regexdf
                                username_box.send_keys(regexdf.strip().decode('utf8'))
                                sleep(0.5)
                                username_box.send_keys(".".decode('utf8'))
                             
    #                                sleep(1)
    #                                transbox=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div/div/div[2]/div[2]/div/div/div')
    #                                transbox.click()
                                counter+=1
    
                                sleep(random.randint(3,4))
                                
    
    
                                trans_text=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[3]/div[1]/div[2]/div')
                                text5=trans_text.text
    #                                pyperclip.copy("")
    #                                sleep(1.5)
                                text.append(text5)
                                
                                
                            except Exception as e:
                                pass
                            sleep(1.5)
                        text.append(text4)
                        
                        
                        
            else:
                regexdf=""
                if len(text2[k]) > 2:
                    temptext=""
                    text4=""
                    temptxt=re.sub('[A-Za-z0-9 ]+', ' ', text2[k])
                    text4=re.sub('[^A-Za-z0-9 ]+', ' ', text2[k])
                    if len(temptxt.strip()) !=0:
                        regexdf=' '.join(temptxt.split()).lower()
        #                     if k!=0:
                        try:
                            username_box.send_keys(".".decode('utf8'))
                            clearbox.click()
            #                     print regexdf
                            username_box.send_keys(regexdf.strip().decode('utf8'))
                            sleep(0.5)
                            username_box.send_keys(".".decode('utf8'))
                        
                         
    #                            sleep(1)
    #                            transbox=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div/div/div[2]/div[2]/div/div/div')
    #                            transbox.click()
    #                        counter+=1
    
                            sleep(random.randint(3,4))
    
    
                            trans_text=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[3]/div[1]/div[2]/div')
                            text5=trans_text.text
#                                pyperclip.copy("")
#                                sleep(1.5)
                            text.append(text5)
                            
                            
                        except Exception as e:
                            pass
                        sleep(1.5)
                    text.append(text4)
                      
           
            trans=" ".join(text)
        #        if counter>100:
        #            counter=0
        #            sleep(120)
        
        #    print trans
        tmptxt1=" ".join(trans.split())
        file3.iloc[i,2]=tmptxt1
        print i
        print file3.iloc[i,1]
        print file3.iloc[i,2]
#        print file3.iloc[i,1]
#        print file3.iloc[i,2]
    return file3


def translatescrp2(df):
    file3 = pd.DataFrame(index=range(len(df.index)), columns=["id","Meta","Transmeta"])
    file3 = file3.fillna("")
    
    for i in range(len(df.index)):
        file3.iloc[i,0]=df.iloc[i,0]
        file3.iloc[i,1]=df.iloc[i,1]
        
        #    username_box.send_keys(u"हिंदी")
        counter=0
        
        def split(input,size):
            return [input[start:start+size] for start in range(0,len(input),size)]
        #    tmptxt1=range(len(text10))
        #    stop_words=pd.read_csv('C:/Users/Administrator/Desktop/VIDOOLY NILANJAN/Python/Translate/stop_words.csv',encoding='utf-8',header=None)
        #    stop_word=[]
        #    for i in range(len(stop_words)):
        #        stop_word.append(stop_words.iloc[i,0])
            
        #    counter=0
        #    for i in range(len(text10)):
        trans=""
        meta=df.iloc[i,1]
        #meta="كرتون ميكى ماوس بالعربى,طفل ميكي ماوس,جاد وعصومي براعم الجنة,جاد وإياد,طيور الجنة"
        
        #    meta=file1.iloc[6,1]
        text1= meta.replace('_','').replace('`','').replace('“','').replace('”','').replace('$','').replace('¢','').replace('£','').replace('¤','').replace('¥','').replace('֏','').replace('؋','').replace('৲','').replace('৳','').replace('৻','').replace('૱','').replace('௹','').replace('฿','').replace('៛','').replace('₠','').replace('₡','').replace('₢','').replace('₣','').replace('₤','').replace('₥','').replace('₦','').replace('₧','').replace('₨','').replace('₩','').replace('₪','').replace('₫','').replace('€','').replace('₭','').replace('₮','').replace('₯','').replace('₰','').replace('₱','').replace('₲','').replace('₳','').replace('₴','').replace('₵','').replace('₶','').replace('₷','').replace('₸','').replace('₹','').replace('₺','').replace('₻','').replace('₼','').replace('₽','').replace('₾','').replace('₿','').replace('﷼','').replace('﹩','').replace('＄','').replace('￠','').replace('￡','').replace('￥','').replace('￦','').replace("&","").replace("'","").replace("°","").replace("’","").replace("🥇","").replace("🥈","").replace("🥉","").replace("🆎","").replace("🏧","").replace("🅰","").replace("🇦🇫","").replace("🇦🇱","").replace("🇩🇿","").replace("🇦🇸","").replace("🇦🇩","").replace("🇦🇴","").replace("🇦🇮","").replace("🇦🇶","").replace("🇦🇬","").replace("♒","").replace("🇦🇷","").replace("♈","").replace("🇦🇲","").replace("🇦🇼","").replace("🇦🇨","").replace("🇦🇺","").replace("🇦🇹","").replace("🇦🇿","").replace("🔙","").replace("🅱","").replace("🇧🇸","").replace("🇧🇭","").replace("🇧🇩","").replace("🇧🇧","").replace("🇧🇾","").replace("🇧🇪","").replace("🇧🇿","").replace("🇧🇯","").replace("🇧🇲","").replace("🇧🇹","").replace("🇧🇴","").replace("🇧🇦","").replace("🇧🇼","").replace("🇧🇻","").replace("🇧🇷","").replace("🇮🇴","").replace("🇻🇬","").replace("🇧🇳","").replace("🇧🇬","").replace("🇧🇫","").replace("🇧🇮","").replace("🆑","").replace("🆒","").replace("🇰🇭","").replace("🇨🇲","").replace("🇨🇦","").replace("🇮🇨","").replace("♋","").replace("🇨🇻","").replace("♑","").replace("🇧🇶","").replace("🇰🇾","").replace("🇨🇫","").replace("🇪🇦","").replace("🇹🇩","").replace("🇨🇱","").replace("🇨🇳","").replace("🇨🇽","").replace("🎄","").replace("🇨🇵","").replace("🇨🇨","").replace("🇨🇴","").replace("🇰🇲","").replace("🇨🇬","").replace("🇨🇩","").replace("🇨🇰","").replace("🇨🇷","").replace("🇭🇷","").replace("🇨🇺","").replace("🇨🇼","").replace("🇨🇾","").replace("🇨🇿","").replace("🇨🇮","").replace("🇩🇰","").replace("🇩🇬","").replace("🇩🇯","").replace("🇩🇲","").replace("🇩🇴","").replace("🔚","").replace("🇪🇨","").replace("🇪🇬","").replace("🇸🇻","").replace("🏴󠁧󠁢󠁥󠁮󠁧󠁿","").replace("🇬🇶","").replace("🇪🇷","").replace("🇪🇪","").replace("🇪🇹","").replace("🇪🇺","").replace("🆓","").replace("🇫🇰","").replace("🇫🇴","").replace("🇫🇯","").replace("🇫🇮","").replace("🇫🇷","").replace("🇬🇫","").replace("🇵🇫","").replace("🇹🇫","").replace("🇬🇦","").replace("🇬🇲","").replace("♊","").replace("🇬🇪","").replace("🇩🇪","").replace("🇬🇭","").replace("🇬🇮","").replace("🇬🇷","").replace("🇬🇱","").replace("🇬🇩","").replace("🇬🇵","").replace("🇬🇺","").replace("🇬🇹","").replace("🇬🇬","").replace("🇬🇳","").replace("🇬🇼","").replace("🇬🇾","").replace("🇭🇹","").replace("🇭🇲","").replace("🇭🇳","").replace("🇭🇰","").replace("🇭🇺","").replace("🆔","").replace("🇮🇸","").replace("🇮🇳","").replace("🇮🇩","").replace("🇮🇷","").replace("🇮🇶","").replace("🇮🇪","").replace("🇮🇲","").replace("🇮🇱","").replace("🇮🇹","").replace("🇯🇲","").replace("🇯🇵","").replace("🉑","").replace("🈸","").replace("🉐","").replace("🏯","").replace("㊗","").replace("🈹","").replace("🎎","").replace("🈚","").replace("🈁","").replace("🈷","").replace("🈵","").replace("🈶","").replace("🈺","").replace("🈴","").replace("🏣","").replace("🈲","").replace("🈯","").replace("㊙","").replace("🈂","").replace("🔰","").replace("🈳","").replace("🇯🇪","").replace("🇯🇴","").replace("🇰🇿","").replace("🇰🇪","").replace("🇰🇮","").replace("🇽🇰","").replace("🇰🇼","").replace("🇰🇬","").replace("🇱🇦","").replace("🇱🇻","").replace("🇱🇧","").replace("♌","").replace("🇱🇸","").replace("🇱🇷","").replace("♎","").replace("🇱🇾","").replace("🇱🇮","").replace("🇱🇹","").replace("🇱🇺","").replace("🇲🇴","").replace("🇲🇰","").replace("🇲🇬","").replace("🇲🇼","").replace("🇲🇾","").replace("🇲🇻","").replace("🇲🇱","").replace("🇲🇹","").replace("🇲🇭","").replace("🇲🇶","").replace("🇲🇷","").replace("🇲🇺","").replace("🇾🇹","").replace("🇲🇽","").replace("🇫🇲","").replace("🇲🇩","").replace("🇲🇨","").replace("🇲🇳","").replace("🇲🇪","").replace("🇲🇸","").replace("🇲🇦","").replace("🇲🇿","").replace("🤶","").replace("🤶🏿","").replace("🤶🏻","").replace("🤶🏾","").replace("🤶🏼","").replace("🤶🏽","").replace("🇲🇲","").replace("🆕","").replace("🆖","").replace("🇳🇦","").replace("🇳🇷","").replace("🇳🇵","").replace("🇳🇱","").replace("🇳🇨","").replace("🇳🇿","").replace("🇳🇮","").replace("🇳🇪","").replace("🇳🇬","").replace("🇳🇺","").replace("🇳🇫","").replace("🇰🇵","").replace("🇲🇵","").replace("🇳🇴","").replace("🆗","").replace("👌","").replace("👌🏿","").replace("👌🏻","").replace("👌🏾","").replace("👌🏼","").replace("👌🏽","").replace("🔛","").replace("🅾","").replace("🇴🇲","").replace("⛎","").replace("🅿","").replace("🇵🇰","").replace("🇵🇼","").replace("🇵🇸","").replace("🇵🇦","").replace("🇵🇬","").replace("🇵🇾","").replace("🇵🇪","").replace("🇵🇭","").replace("♓","").replace("🇵🇳","").replace("🇵🇱","").replace("🇵🇹","").replace("🇵🇷","").replace("🇶🇦","").replace("🇷🇴","").replace("🇷🇺","").replace("🇷🇼","").replace("🇷🇪","").replace("🔜","").replace("🆘","").replace("♐","").replace("🇼🇸","").replace("🇸🇲","").replace("🎅","").replace("🎅🏿","").replace("🎅🏻","").replace("🎅🏾","").replace("🎅🏼","").replace("🎅🏽","").replace("🇸🇦","").replace("♏","").replace("🏴󠁧󠁢󠁳󠁣󠁴󠁿","").replace("🇸🇳","").replace("🇷🇸","").replace("🇸🇨","").replace("🇸🇱","").replace("🇸🇬","").replace("🇸🇽","").replace("🇸🇰","").replace("🇸🇮","").replace("🇸🇧","").replace("🇸🇴","").replace("🇿🇦","").replace("🇬🇸","").replace("🇰🇷","").replace("🇸🇸","").replace("🇪🇸","").replace("🇱🇰","").replace("🇧🇱","").replace("🇸🇭","").replace("🇰🇳","").replace("🇱🇨","").replace("🇲🇫","").replace("🇵🇲","").replace("🇻🇨","").replace("🗽","").replace("🇸🇩","").replace("🇸🇷","").replace("🇸🇯","").replace("🇸🇿","").replace("🇸🇪","").replace("🇨🇭","").replace("🇸🇾","").replace("🇸🇹","").replace("🦖","").replace("🔝","").replace("🇹🇼","").replace("🇹🇯","").replace("🇹🇿","").replace("♉","").replace("🇹🇭","").replace("🇹🇱","").replace("🇹🇬","").replace("🇹🇰","").replace("🗼","").replace("🇹🇴","").replace("🇹🇹","").replace("🇹🇦","").replace("🇹🇳","").replace("🇹🇷","").replace("🇹🇲","").replace("🇹🇨","").replace("🇹🇻","").replace("🇺🇲","").replace("🇻🇮","").replace("🆙","").replace("🇺🇬","").replace("🇺🇦","").replace("🇦🇪","").replace("🇬🇧","").replace("🇺🇳","").replace("🇺🇸","").replace("🇺🇾","").replace("🇺🇿","").replace("🆚","").replace("🇻🇺","").replace("🇻🇦","").replace("🇻🇪","").replace("🇻🇳","").replace("♍","").replace("🏴󠁧󠁢󠁷󠁬󠁳󠁿","").replace("🇼🇫","").replace("🇪🇭","").replace("🇾🇪","").replace("🇿🇲","").replace("🇿🇼","").replace("🎟","").replace("🧑","").replace("🧑🏿","").replace("🧑🏻","").replace("🧑🏾","").replace("🧑🏼","").replace("🧑🏽","").replace("🚡","").replace("✈","").replace("🛬","").replace("🛫","").replace("⏰","").replace("⚗","").replace("👽","").replace("👾","").replace("🚑","").replace("🏈","").replace("🏺","").replace("⚓","").replace("💢","").replace("😠","").replace("👿","").replace("😧","").replace("🐜","").replace("📶","").replace("😰","").replace("🚛","").replace("🎨","").replace("😲","").replace("⚛","").replace("🚗","").replace("🥑","").replace("👶","").replace("👼","").replace("👼🏿","").replace("👼🏻","").replace("👼🏾","").replace("👼🏼","").replace("👼🏽","").replace("🍼","").replace("🐤","").replace("👶🏿","").replace("👶🏻","").replace("👶🏾","").replace("👶🏼","").replace("👶🏽","").replace("🚼","").replace("👇","").replace("👇🏿","").replace("👇🏻","").replace("👇🏾","").replace("👇🏼","").replace("👇🏽","").replace("👈","").replace("👈🏿","").replace("👈🏻","").replace("👈🏾","").replace("👈🏼","").replace("👈🏽","").replace("👉","").replace("👉🏿","").replace("👉🏻","").replace("👉🏾","").replace("👉🏼","").replace("👉🏽","").replace("👆","").replace("👆🏿","").replace("👆🏻","").replace("👆🏾","").replace("👆🏼","").replace("👆🏽","").replace("🥓","").replace("🏸","").replace("🛄","").replace("🥖","").replace("⚖","").replace("🎈","").replace("🗳","").replace("☑","").replace("🍌","").replace("🏦","").replace("📊","").replace("💈","").replace("⚾","").replace("🏀","").replace("🦇","").replace("🛁","").replace("🔋","").replace("🏖","").replace("😁","").replace("🐻","").replace("🧔","").replace("🧔🏿","").replace("🧔🏻","").replace("🧔🏾","").replace("🧔🏼","").replace("🧔🏽","").replace("💓","").replace("🛏","").replace("🍺","").replace("🔔","").replace("🔕","").replace("🛎","").replace("🍱","").replace("🚲","").replace("👙","").replace("🧢","").replace("☣","").replace("🐦","").replace("🎂","").replace("⚫","").replace("🏴","").replace("🖤","").replace("⬛","").replace("◾","").replace("◼","").replace("✒","").replace("▪","").replace("🔲","").replace("👱‍♂️","").replace("👱🏿‍♂️","").replace("👱🏻‍♂️","").replace("👱🏾‍♂️","").replace("👱🏼‍♂️","").replace("👱🏽‍♂️","").replace("👱","").replace("👱🏿","").replace("👱🏻","").replace("👱🏾","").replace("👱🏼","").replace("👱🏽","").replace("👱‍♀️","").replace("👱🏿‍♀️","").replace("👱🏻‍♀️","").replace("👱🏾‍♀️","").replace("👱🏼‍♀️","").replace("👱🏽‍♀️","").replace("🌼","").replace("🐡","").replace("📘","").replace("🔵","").replace("💙","").replace("🐗","").replace("💣","").replace("🔖","").replace("📑","").replace("📚","").replace("🍾","").replace("💐","").replace("🏹","").replace("🥣","").replace("🎳","").replace("🥊","").replace("👦","").replace("👦🏿","").replace("👦🏻","").replace("👦🏾","").replace("👦🏼","").replace("👦🏽","").replace("🧠","").replace("🍞","").replace("🤱","").replace("🤱🏿","").replace("🤱🏻","").replace("🤱🏾","").replace("🤱🏼","").replace("🤱🏽","").replace("👰","").replace("👰🏿","").replace("👰🏻","").replace("👰🏾","").replace("👰🏼","").replace("👰🏽","").replace("🌉","").replace("💼","").replace("🔆","").replace("🥦","").replace("💔","").replace("🐛","").replace("🏗","").replace("🚅","").replace("🌯","").replace("🚌","").replace("🚏","").replace("👤","").replace("👥","").replace("🦋","").replace("🌵","").replace("📆","").replace("🤙","").replace("🤙🏿","").replace("🤙🏻","").replace("🤙🏾","").replace("🤙🏼","").replace("🤙🏽","").replace("🐫","").replace("📷","").replace("📸","").replace("🏕","").replace("🕯","").replace("🍬","").replace("🥫","").replace("🛶","").replace("🗃","").replace("📇","").replace("🗂","").replace("🎠","").replace("🎏","").replace("🥕","").replace("🏰","").replace("🐱","").replace("🐱","").replace("😹","").replace("😼","").replace("⛓","").replace("📉","").replace("📈","").replace("💹","").replace("🧀","").replace("🏁","").replace("🍒","").replace("🌸","").replace("🌰","").replace("🐔","").replace("🧒","").replace("🧒🏿","").replace("🧒🏻","").replace("🧒🏾","").replace("🧒🏼","").replace("🧒🏽","").replace("🚸","").replace("🐿","").replace("🍫","").replace("🥢","").replace("⛪","").replace("🚬","").replace("🎦","").replace("Ⓜ","").replace("🎪","").replace("🏙","").replace("🌆","").replace("🗜","").replace("🎬","").replace("👏","").replace("👏🏿","").replace("👏🏻","").replace("👏🏾","").replace("👏🏼","").replace("👏🏽","").replace("🏛","").replace("🍻","").replace("🥂","").replace("📋","").replace("🔃","").replace("📕","").replace("📪","").replace("📫","").replace("🌂","").replace("☁","").replace("🌩","").replace("⛈","").replace("🌧","").replace("🌨","").replace("🤡","").replace("♣","").replace("👝","").replace("🧥","").replace("🍸","").replace("🥥","").replace("⚰","").replace("💥","").replace("☄","").replace("💽","").replace("🖱","").replace("🎊","").replace("😖","").replace("😕","").replace("🚧","").replace("👷","").replace("👷🏿","").replace("👷🏻","").replace("👷🏾","").replace("👷🏼","").replace("👷🏽","").replace("🎛","").replace("🏪","").replace("🍚","").replace("🍪","").replace("🍳","").replace("©","").replace("🛋","").replace("🔄","").replace("💑","").replace("👨‍❤️‍👨","").replace("👩‍❤️‍👨","").replace("👩‍❤️‍👩","").replace("🐮","").replace("🐮","").replace("🤠","").replace("🦀","").replace("🖍","").replace("💳","").replace("🌙","").replace("🦗","").replace("🏏","").replace("🐊","").replace("🥐","").replace("❌","").replace("❎","").replace("🤞","").replace("🤞🏿","").replace("🤞🏻","").replace("🤞🏾","").replace("🤞🏼","").replace("🤞🏽","").replace("🎌","").replace("⚔","").replace("👑","").replace("😿","").replace("😢","").replace("🔮","").replace("🥒","").replace("🥤","").replace("🥌","").replace("➰","").replace("💱","").replace("🍛","").replace("🍮","").replace("🛃","").replace("🥩","").replace("🌀","").replace("🗡","").replace("🍡","").replace("💨","").replace("🌳","").replace("🦌","").replace("🚚","").replace("🏬","").replace("🏚","").replace("🏜","").replace("🏝","").replace("🖥","").replace("🕵","").replace("🕵🏿","").replace("🕵🏻","").replace("🕵🏾","").replace("🕵🏼","").replace("🕵🏽","").replace("♦","").replace("💠","").replace("🔅","").replace("🎯","").replace("😞","").replace("💫","").replace("😵","").replace("🐶","").replace("🐶","").replace("💵","").replace("🐬","").replace("🚪","").replace("🔯","").replace("➿","").replace("‼","").replace("🍩","").replace("🕊","").replace("↙","").replace("↘","").replace("⬇","").replace("😓","").replace("🔽","").replace("🐉","").replace("🐲","").replace("👗","").replace("🤤","").replace("💧","").replace("🥁","").replace("🦆","").replace("🥟","").replace("📀","").replace("📧","").replace("🦅","").replace("👂","").replace("👂🏿","").replace("👂🏻","").replace("👂🏾","").replace("👂🏼","").replace("👂🏽","").replace("🌽","").replace("🍳","").replace("🍆","").replace("✴","").replace("✳","").replace("🕣","").replace("🕗","").replace("⏏","").replace("🔌","").replace("🐘","").replace("🕦","").replace("🕚","").replace("🧝","").replace("🧝🏿","").replace("🧝🏻","").replace("🧝🏾","").replace("🧝🏼","").replace("🧝🏽","").replace("✉","").replace("📩","").replace("💶","").replace("🌲","").replace("🐑","").replace("❗","").replace("⁉","").replace("🤯","").replace("😑","").replace("👁","").replace("👁️‍🗨️","").replace("👀","").replace("😘","").replace("😋","").replace("😱","").replace("🤮","").replace("🤭","").replace("🤕","").replace("😷","").replace("🧐","").replace("😮","").replace("🤨","").replace("🙄","").replace("😤","").replace("🤬","").replace("😂","").replace("🤒","").replace("😛","").replace("😶","").replace("🏭","").replace("🧚","").replace("🧚🏿","").replace("🧚🏻","").replace("🧚🏾","").replace("🧚🏼","").replace("🧚🏽","").replace("🍂","").replace("👪","").replace("👨‍👦","").replace("👨‍👦‍👦","").replace("👨‍👧","").replace("👨‍👧‍👦","").replace("👨‍👧‍👧","").replace("👨‍👨‍👦","").replace("👨‍👨‍👦‍👦","").replace("👨‍👨‍👧","").replace("👨‍👨‍👧‍👦","").replace("👨‍👨‍👧‍👧","").replace("👨‍👩‍👦","").replace("👨‍👩‍👦‍👦","").replace("👨‍👩‍👧","").replace("👨‍👩‍👧‍👦","").replace("👨‍👩‍👧‍👧","").replace("👩‍👦","").replace("👩‍👦‍👦","").replace("👩‍👧","").replace("👩‍👧‍👦","").replace("👩‍👧‍👧","").replace("👩‍👩‍👦","").replace("👩‍👩‍👦‍👦","").replace("👩‍👩‍👧","").replace("👩‍👩‍👧‍👦","").replace("👩‍👩‍👧‍👧","").replace("⏩","").replace("⏬","").replace("⏪","").replace("⏫","").replace("📠","").replace("😨","").replace("♀","").replace("🎡","").replace("⛴","").replace("🏑","").replace("🗄","").replace("📁","").replace("🎞","").replace("📽","").replace("🔥","").replace("🚒","").replace("🎆","").replace("🌓","").replace("🌛","").replace("🐟","").replace("🍥","").replace("🎣","").replace("🕠","").replace("🕔","").replace("⛳","").replace("🔦","").replace("⚜","").replace("💪","").replace("💪🏿","").replace("💪🏻","").replace("💪🏾","").replace("💪🏼","").replace("💪🏽","").replace("💾","").replace("🎴","").replace("😳","").replace("🛸","").replace("🌫","").replace("🌁","").replace("🙏","").replace("🙏🏿","").replace("🙏🏻","").replace("🙏🏾","").replace("🙏🏼","").replace("🙏🏽","").replace("👣","").replace("🍴","").replace("🍽","").replace("🥠","").replace("⛲","").replace("🖋","").replace("🕟","").replace("🍀","").replace("🕓","").replace("🦊","").replace("🖼","").replace("🍟","").replace("🍤","").replace("🐸","").replace("🐥","").replace("☹","").replace("😦","").replace("⛽","").replace("🌕","").replace("🌝","").replace("⚱","").replace("🎲","").replace("⚙","").replace("💎","").replace("🧞","").replace("👻","").replace("🦒","").replace("👧","").replace("👧🏿","").replace("👧🏻","").replace("👧🏾","").replace("👧🏼","").replace("👧🏽","").replace("🥛","").replace("👓","").replace("🌎","").replace("🌏","").replace("🌍","").replace("🌐","").replace("🧤","").replace("🌟","").replace("🥅","").replace("🐐","").replace("👺","").replace("🦍","").replace("🎓","").replace("🍇","").replace("🍏","").replace("📗","").replace("💚","").replace("🥗","").replace("😬","").replace("😺","").replace("😸","").replace("😀","").replace("😃","").replace("😄","").replace("😅","").replace("😆","").replace("💗","").replace("💂","").replace("💂🏿","").replace("💂🏻","").replace("💂🏾","").replace("💂🏼","").replace("💂🏽","").replace("🎸","").replace("🍔","").replace("🔨","").replace("⚒","").replace("🛠","").replace("🐹","").replace("🖐","").replace("🖐🏿","").replace("🖐🏻","").replace("🖐🏾","").replace("🖐🏼","").replace("🖐🏽","").replace("👜","").replace("🤝","").replace("🐣","").replace("🎧","").replace("🙉","").replace("💟","").replace("♥","").replace("💘","").replace("💝","").replace("✔","").replace("➗","").replace("💲","").replace("❣","").replace("⭕","").replace("➖","").replace("✖","").replace("➕","").replace("🦔","").replace("🚁","").replace("🌿","").replace("🌺","").replace("👠","").replace("🚄","").replace("⚡","").replace("🕳","").replace("🍯","").replace("🐝","").replace("🚥","").replace("🐴","").replace("🐴","").replace("🏇","").replace("🏇🏿","").replace("🏇🏻","").replace("🏇🏾","").replace("🏇🏼","").replace("🏇🏽","").replace("🏥","").replace("☕","").replace("🌭","").replace("🌶","").replace("♨","").replace("🏨","").replace("⌛","").replace("⏳","").replace("🏠","").replace("🏡","").replace("🏘","").replace("🤗","").replace("💯","").replace("😯","").replace("🍨","").replace("🏒","").replace("⛸","").replace("📥","").replace("📨","").replace("☝","").replace("☝🏿","").replace("☝🏻","").replace("☝🏾","").replace("☝🏼","").replace("☝🏽","").replace("ℹ","").replace("🔤","").replace("🔡","").replace("🔠","").replace("🔢","").replace("🔣","").replace("🎃","").replace("👖","").replace("🃏","").replace("🕹","").replace("🕋","").replace("🔑","").replace("⌨","").replace("#️⃣","").replace("*️⃣","").replace("0️⃣","").replace("1️⃣","").replace("🔟","").replace("2️⃣","").replace("3️⃣","").replace("4️⃣","").replace("5️⃣","").replace("6️⃣","").replace("7️⃣","").replace("8️⃣","").replace("9️⃣","").replace("🛴","").replace("👘","").replace("💋","").replace("👨‍❤️‍💋‍👨","").replace("💋","").replace("👩‍❤️‍💋‍👨","").replace("👩‍❤️‍💋‍👩","").replace("😽","").replace("😗","").replace("😚","").replace("😙","").replace("🔪","").replace("🥝","").replace("🐨","").replace("🏷","").replace("🐞","").replace("💻","").replace("🔷","").replace("🔶","").replace("🌗","").replace("🌜","").replace("⏮","").replace("✝","").replace("🍃","").replace("📒","").replace("🤛","").replace("🤛🏿","").replace("🤛🏻","").replace("🤛🏾","").replace("🤛🏼","").replace("🤛🏽","").replace("↔","").replace("⬅","").replace("↪","").replace("🛅","").replace("🗨","").replace("🍋","").replace("🐆","").replace("🎚","").replace("💡","").replace("🚈","").replace("🔗","").replace("🖇","").replace("🦁","").replace("💄","").replace("🚮","").replace("🦎","").replace("🔒","").replace("🔐","").replace("🔏","").replace("🚂","").replace("🍭","").replace("😭","").replace("📢","").replace("🤟","").replace("🤟🏿","").replace("🤟🏻","").replace("🤟🏾","").replace("🤟🏼","").replace("🤟🏽","").replace("🏩","").replace("💌","").replace("🤥","").replace("🧙","").replace("🧙🏿","").replace("🧙🏻","").replace("🧙🏾","").replace("🧙🏼","").replace("🧙🏽","").replace("🔍","").replace("🔎","").replace("🀄","").replace("♂","").replace("👨","").replace("👫","").replace("👨‍🎨","").replace("👨🏿‍🎨","").replace("👨🏻‍🎨","").replace("👨🏾‍🎨","").replace("👨🏼‍🎨","").replace("👨🏽‍🎨","").replace("👨‍🚀","").replace("👨🏿‍🚀","").replace("👨🏻‍🚀","").replace("👨🏾‍🚀","").replace("👨🏼‍🚀","").replace("👨🏽‍🚀","").replace("🚴‍♂️","").replace("🚴🏿‍♂️","").replace("🚴🏻‍♂️","").replace("🚴🏾‍♂️","").replace("🚴🏼‍♂️","").replace("🚴🏽‍♂️","").replace("⛹️‍♂️","").replace("⛹🏿‍♂️","").replace("⛹🏻‍♂️","").replace("⛹🏾‍♂️","").replace("⛹🏼‍♂️","").replace("⛹🏽‍♂️","").replace("🙇‍♂️","").replace("🙇🏿‍♂️","").replace("🙇🏻‍♂️","").replace("🙇🏾‍♂️","").replace("🙇🏼‍♂️","").replace("🙇🏽‍♂️","").replace("🤸‍♂️","").replace("🤸🏿‍♂️","").replace("🤸🏻‍♂️","").replace("🤸🏾‍♂️","").replace("🤸🏼‍♂️","").replace("🤸🏽‍♂️","").replace("🧗‍♂️","").replace("🧗🏿‍♂️","").replace("🧗🏻‍♂️","").replace("🧗🏾‍♂️","").replace("🧗🏼‍♂️","").replace("🧗🏽‍♂️","").replace("👷‍♂️","").replace("👷🏿‍♂️","").replace("👷🏻‍♂️","").replace("👷🏾‍♂️","").replace("👷🏼‍♂️","").replace("👷🏽‍♂️","").replace("👨‍🍳","").replace("👨🏿‍🍳","").replace("👨🏻‍🍳","").replace("👨🏾‍🍳","").replace("👨🏼‍🍳","").replace("👨🏽‍🍳","").replace("🕺","").replace("🕺🏿","").replace("🕺🏻","").replace("🕺🏾","").replace("🕺🏼","").replace("🕺🏽","").replace("👨🏿","").replace("🕵️‍♂️","").replace("🕵🏿‍♂️","").replace("🕵🏻‍♂️","").replace("🕵🏾‍♂️","").replace("🕵🏼‍♂️","").replace("🕵🏽‍♂️","").replace("🧝‍♂️","").replace("🧝🏿‍♂️","").replace("🧝🏻‍♂️","").replace("🧝🏾‍♂️","").replace("🧝🏼‍♂️","").replace("🧝🏽‍♂️","").replace("🤦‍♂️","").replace("🤦🏿‍♂️","").replace("🤦🏻‍♂️","").replace("🤦🏾‍♂️","").replace("🤦🏼‍♂️","").replace("🤦🏽‍♂️","").replace("👨‍🏭","").replace("👨🏿‍🏭","").replace("👨🏻‍🏭","").replace("👨🏾‍🏭","").replace("👨🏼‍🏭","").replace("👨🏽‍🏭","").replace("🧚‍♂️","").replace("🧚🏿‍♂️","").replace("🧚🏻‍♂️","").replace("🧚🏾‍♂️","").replace("🧚🏼‍♂️","").replace("🧚🏽‍♂️","").replace("👨‍🌾","").replace("👨🏿‍🌾","").replace("👨🏻‍🌾","").replace("👨🏾‍🌾","").replace("👨🏼‍🌾","").replace("👨🏽‍🌾","").replace("👨‍🚒","").replace("👨🏿‍🚒","").replace("👨🏻‍🚒","").replace("👨🏾‍🚒","").replace("👨🏼‍🚒","").replace("👨🏽‍🚒","").replace("🙍‍♂️","").replace("🙍🏿‍♂️","").replace("🙍🏻‍♂️","").replace("🙍🏾‍♂️","").replace("🙍🏼‍♂️","").replace("🙍🏽‍♂️","").replace("🧞‍♂️","").replace("🙅‍♂️","").replace("🙅🏿‍♂️","").replace("🙅🏻‍♂️","").replace("🙅🏾‍♂️","").replace("🙅🏼‍♂️","").replace("🙅🏽‍♂️","").replace("🙆‍♂️","").replace("🙆🏿‍♂️","").replace("🙆🏻‍♂️","").replace("🙆🏾‍♂️","").replace("🙆🏼‍♂️","").replace("🙆🏽‍♂️","").replace("💇‍♂️","").replace("💇🏿‍♂️","").replace("💇🏻‍♂️","").replace("💇🏾‍♂️","").replace("💇🏼‍♂️","").replace("💇🏽‍♂️","").replace("💆‍♂️","").replace("💆🏿‍♂️","").replace("💆🏻‍♂️","").replace("💆🏾‍♂️","").replace("💆🏼‍♂️","").replace("💆🏽‍♂️","").replace("🏌️‍♂️","").replace("🏌🏿‍♂️","").replace("🏌🏻‍♂️","").replace("🏌🏾‍♂️","").replace("🏌🏼‍♂️","").replace("🏌🏽‍♂️","").replace("💂‍♂️","").replace("💂🏿‍♂️","").replace("💂🏻‍♂️","").replace("💂🏾‍♂️","").replace("💂🏼‍♂️","").replace("💂🏽‍♂️","").replace("👨‍⚕️","").replace("👨🏿‍⚕️","").replace("👨🏻‍⚕️","").replace("👨🏾‍⚕️","").replace("👨🏼‍⚕️","").replace("👨🏽‍⚕️","").replace("🧘‍♂️","").replace("🧘🏿‍♂️","").replace("🧘🏻‍♂️","").replace("🧘🏾‍♂️","").replace("🧘🏼‍♂️","").replace("🧘🏽‍♂️","").replace("🧖‍♂️","").replace("🧖🏿‍♂️","").replace("🧖🏻‍♂️","").replace("🧖🏾‍♂️","").replace("🧖🏼‍♂️","").replace("🧖🏽‍♂️","").replace("🕴","").replace("🕴🏿","").replace("🕴🏻","").replace("🕴🏾","").replace("🕴🏼","").replace("🕴🏽","").replace("🤵","").replace("🤵🏿","").replace("🤵🏻","").replace("🤵🏾","").replace("🤵🏼","").replace("🤵🏽","").replace("👨‍⚖️","").replace("👨🏿‍⚖️","").replace("👨🏻‍⚖️","").replace("👨🏾‍⚖️","").replace("👨🏼‍⚖️","").replace("👨🏽‍⚖️","").replace("🤹‍♂️","").replace("🤹🏿‍♂️","").replace("🤹🏻‍♂️","").replace("🤹🏾‍♂️","").replace("🤹🏼‍♂️","").replace("🤹🏽‍♂️","").replace("🏋️‍♂️","").replace("🏋🏿‍♂️","").replace("🏋🏻‍♂️","").replace("🏋🏾‍♂️","").replace("🏋🏼‍♂️","").replace("🏋🏽‍♂️","").replace("👨🏻","").replace("🧙‍♂️","").replace("🧙🏿‍♂️","").replace("🧙🏻‍♂️","").replace("🧙🏾‍♂️","").replace("🧙🏼‍♂️","").replace("🧙🏽‍♂️","").replace("👨‍🔧","").replace("👨🏿‍🔧","").replace("👨🏻‍🔧","").replace("👨🏾‍🔧","").replace("👨🏼‍🔧","").replace("👨🏽‍🔧","").replace("👨🏾","").replace("👨🏼","").replace("👨🏽","").replace("🚵‍♂️","").replace("🚵🏿‍♂️","").replace("🚵🏻‍♂️","").replace("🚵🏾‍♂️","").replace("🚵🏼‍♂️","").replace("🚵🏽‍♂️","").replace("👨‍💼","").replace("👨🏿‍💼","").replace("👨🏻‍💼","").replace("👨🏾‍💼","").replace("👨🏼‍💼","").replace("👨🏽‍💼","").replace("👨‍✈️","").replace("👨🏿‍✈️","").replace("👨🏻‍✈️","").replace("👨🏾‍✈️","").replace("👨🏼‍✈️","").replace("👨🏽‍✈️","").replace("🤾‍♂️","").replace("🤾🏿‍♂️","").replace("🤾🏻‍♂️","").replace("🤾🏾‍♂️","").replace("🤾🏼‍♂️","").replace("🤾🏽‍♂️","").replace("🤽‍♂️","").replace("🤽🏿‍♂️","").replace("🤽🏻‍♂️","").replace("🤽🏾‍♂️","").replace("🤽🏼‍♂️","").replace("🤽🏽‍♂️","").replace("👮‍♂️","").replace("👮🏿‍♂️","").replace("👮🏻‍♂️","").replace("👮🏾‍♂️","").replace("👮🏼‍♂️","").replace("👮🏽‍♂️","").replace("🙎‍♂️","").replace("🙎🏿‍♂️","").replace("🙎🏻‍♂️","").replace("🙎🏾‍♂️","").replace("🙎🏼‍♂️","").replace("🙎🏽‍♂️","").replace("🙋‍♂️","").replace("🙋🏿‍♂️","").replace("🙋🏻‍♂️","").replace("🙋🏾‍♂️","").replace("🙋🏼‍♂️","").replace("🙋🏽‍♂️","").replace("🚣‍♂️","").replace("🚣🏿‍♂️","").replace("🚣🏻‍♂️","").replace("🚣🏾‍♂️","").replace("🚣🏼‍♂️","").replace("🚣🏽‍♂️","").replace("🏃‍♂️","").replace("🏃🏿‍♂️","").replace("🏃🏻‍♂️","").replace("🏃🏾‍♂️","").replace("🏃🏼‍♂️","").replace("🏃🏽‍♂️","").replace("👨‍🔬","").replace("👨🏿‍🔬","").replace("👨🏻‍🔬","").replace("👨🏾‍🔬","").replace("👨🏼‍🔬","").replace("👨🏽‍🔬","").replace("🤷‍♂️","").replace("🤷🏿‍♂️","").replace("🤷🏻‍♂️","").replace("🤷🏾‍♂️","").replace("🤷🏼‍♂️","").replace("🤷🏽‍♂️","").replace("👨‍🎤","").replace("👨🏿‍🎤","").replace("👨🏻‍🎤","").replace("👨🏾‍🎤","").replace("👨🏼‍🎤","").replace("👨🏽‍🎤","").replace("👨‍🎓","").replace("👨🏿‍🎓","").replace("👨🏻‍🎓","").replace("👨🏾‍🎓","").replace("👨🏼‍🎓","").replace("👨🏽‍🎓","").replace("🏄‍♂️","").replace("🏄🏿‍♂️","").replace("🏄🏻‍♂️","").replace("🏄🏾‍♂️","").replace("🏄🏼‍♂️","").replace("🏄🏽‍♂️","").replace("🏊‍♂️","").replace("🏊🏿‍♂️","").replace("🏊🏻‍♂️","").replace("🏊🏾‍♂️","").replace("🏊🏼‍♂️","").replace("🏊🏽‍♂️","").replace("👨‍🏫","").replace("👨🏿‍🏫","").replace("👨🏻‍🏫","").replace("👨🏾‍🏫","").replace("👨🏼‍🏫","").replace("👨🏽‍🏫","").replace("👨‍💻","").replace("👨🏿‍💻","").replace("👨🏻‍💻","").replace("👨🏾‍💻","").replace("👨🏼‍💻","").replace("👨🏽‍💻","").replace("💁‍♂️","").replace("💁🏿‍♂️","").replace("💁🏻‍♂️","").replace("💁🏾‍♂️","").replace("💁🏼‍♂️","").replace("💁🏽‍♂️","").replace("🧛‍♂️","").replace("🧛🏿‍♂️","").replace("🧛🏻‍♂️","").replace("🧛🏾‍♂️","").replace("🧛🏼‍♂️","").replace("🧛🏽‍♂️","").replace("🚶‍♂️","").replace("🚶🏿‍♂️","").replace("🚶🏻‍♂️","").replace("🚶🏾‍♂️","").replace("🚶🏼‍♂️","").replace("🚶🏽‍♂️","").replace("👳‍♂️","").replace("👳🏿‍♂️","").replace("👳🏻‍♂️","").replace("👳🏾‍♂️","").replace("👳🏼‍♂️","").replace("👳🏽‍♂️","").replace("👲","").replace("👲🏿","").replace("👲🏻","").replace("👲🏾","").replace("👲🏼","").replace("👲🏽","").replace("🧟‍♂️","").replace("🕰","").replace("👞","").replace("🗾","").replace("🍁","").replace("🥋","").replace("🍖","").replace("⚕","").replace("📣","").replace("🍈","").replace("📝","").replace("👯‍♂️","").replace("🤼‍♂️","").replace("🕎","").replace("🚹","").replace("🧜‍♀️","").replace("🧜🏿‍♀️","").replace("🧜🏻‍♀️","").replace("🧜🏾‍♀️","").replace("🧜🏼‍♀️","").replace("🧜🏽‍♀️","").replace("🧜‍♂️","").replace("🧜🏿‍♂️","").replace("🧜🏻‍♂️","").replace("🧜🏾‍♂️","").replace("🧜🏼‍♂️","").replace("🧜🏽‍♂️","").replace("🧜","").replace("🧜🏿","").replace("🧜🏻","").replace("🧜🏾","").replace("🧜🏼","").replace("🧜🏽","").replace("🚇","").replace("🎤","").replace("🔬","").replace("🖕","").replace("🖕🏿","").replace("🖕🏻","").replace("🖕🏾","").replace("🖕🏼","").replace("🖕🏽","").replace("🎖","").replace("🌌","").replace("🚐","").replace("🗿","").replace("📱","").replace("📴","").replace("📲","").replace("🤑","").replace("💰","").replace("💸","").replace("🐒","").replace("🐵","").replace("🚝","").replace("🎑","").replace("🕌","").replace("🛥","").replace("🛵","").replace("🏍","").replace("🛣","").replace("🗻","").replace("⛰","").replace("🚠","").replace("🚞","").replace("🐭","").replace("🐭","").replace("👄","").replace("🎥","").replace("🍄","").replace("🎹","").replace("🎵","").replace("🎶","").replace("🎼","").replace("🔇","").replace("💅","").replace("💅🏿","").replace("💅🏻","").replace("💅🏾","").replace("💅🏼","").replace("💅🏽","").replace("📛","").replace("🏞","").replace("🤢","").replace("👔","").replace("🤓","").replace("😐","").replace("🌑","").replace("🌚","").replace("📰","").replace("⏭","").replace("🌃","").replace("🕤","").replace("🕘","").replace("🚳","").replace("⛔","").replace("🚯","").replace("📵","").replace("🔞","").replace("🚷","").replace("🚭","").replace("🚱","").replace("👃","").replace("👃🏿","").replace("👃🏻","").replace("👃🏾","").replace("👃🏼","").replace("👃🏽","").replace("📓","").replace("📔","").replace("🔩","").replace("🐙","").replace("🍢","").replace("🏢","").replace("👹","").replace("🛢","").replace("🗝","").replace("👴","").replace("👴🏿","").replace("👴🏻","").replace("👴🏾","").replace("👴🏼","").replace("👴🏽","").replace("👵","").replace("👵🏿","").replace("👵🏻","").replace("👵🏾","").replace("👵🏼","").replace("👵🏽","").replace("🧓","").replace("🧓🏿","").replace("🧓🏻","").replace("🧓🏾","").replace("🧓🏼","").replace("🧓🏽","").replace("🕉","").replace("🚘","").replace("🚍","").replace("👊","").replace("👊🏿","").replace("👊🏻","").replace("👊🏾","").replace("👊🏼","").replace("👊🏽","").replace("🚔","").replace("🚖","").replace("🕜","").replace("🕐","").replace("📖","").replace("📂","").replace("👐","").replace("👐🏿","").replace("👐🏻","").replace("👐🏾","").replace("👐🏼","").replace("👐🏽","").replace("📭","").replace("📬","").replace("💿","").replace("📙","").replace("🧡","").replace("☦","").replace("📤","").replace("🦉","").replace("🐂","").replace("📦","").replace("📄","").replace("📃","").replace("📟","").replace("🖌","").replace("🌴","").replace("🤲","").replace("🤲🏿","").replace("🤲🏻","").replace("🤲🏾","").replace("🤲🏼","").replace("🤲🏽","").replace("🥞","").replace("🐼","").replace("📎","").replace("〽","").replace("🎉","").replace("🛳","").replace("🛂","").replace("⏸","").replace("🐾","").replace("☮","").replace("🍑","").replace("🥜","").replace("🍐","").replace("🖊","").replace("📝","").replace("🐧","").replace("😔","").replace("👯","").replace("🤼","").replace("🎭","").replace("😣","").replace("🚴","").replace("🚴🏿","").replace("🚴🏻","").replace("🚴🏾","").replace("🚴🏼","").replace("🚴🏽","").replace("⛹","").replace("⛹🏿","").replace("⛹🏻","").replace("⛹🏾","").replace("⛹🏼","").replace("⛹🏽","").replace("🙇","").replace("🙇🏿","").replace("🙇🏻","").replace("🙇🏾","").replace("🙇🏼","").replace("🙇🏽","").replace("🤸","").replace("🤸🏿","").replace("🤸🏻","").replace("🤸🏾","").replace("🤸🏼","").replace("🤸🏽","").replace("🧗","").replace("🧗🏿","").replace("🧗🏻","").replace("🧗🏾","").replace("🧗🏼","").replace("🧗🏽","").replace("🤦","").replace("🤦🏿","").replace("🤦🏻","").replace("🤦🏾","").replace("🤦🏼","").replace("🤦🏽","").replace("🤺","").replace("🙍","").replace("🙍🏿","").replace("🙍🏻","").replace("🙍🏾","").replace("🙍🏼","").replace("🙍🏽","").replace("🙅","").replace("🙅🏿","").replace("🙅🏻","").replace("🙅🏾","").replace("🙅🏼","").replace("🙅🏽","").replace("🙆","").replace("🙆🏿","").replace("🙆🏻","").replace("🙆🏾","").replace("🙆🏼","").replace("🙆🏽","").replace("💇","").replace("💇🏿","").replace("💇🏻","").replace("💇🏾","").replace("💇🏼","").replace("💇🏽","").replace("💆","").replace("💆🏿","").replace("💆🏻","").replace("💆🏾","").replace("💆🏼","").replace("💆🏽","").replace("🏌","").replace("🏌🏿","").replace("🏌🏻","").replace("🏌🏾","").replace("🏌🏼","").replace("🏌🏽","").replace("🛌","").replace("🛌🏿","").replace("🛌🏻","").replace("🛌🏾","").replace("🛌🏼","").replace("🛌🏽","").replace("🧘","").replace("🧘🏿","").replace("🧘🏻","").replace("🧘🏾","").replace("🧘🏼","").replace("🧘🏽","").replace("🧖","").replace("🧖🏿","").replace("🧖🏻","").replace("🧖🏾","").replace("🧖🏼","").replace("🧖🏽","").replace("🤹","").replace("🤹🏿","").replace("🤹🏻","").replace("🤹🏾","").replace("🤹🏼","").replace("🤹🏽","").replace("🏋","").replace("🏋🏿","").replace("🏋🏻","").replace("🏋🏾","").replace("🏋🏼","").replace("🏋🏽","").replace("🚵","").replace("🚵🏿","").replace("🚵🏻","").replace("🚵🏾","").replace("🚵🏼","").replace("🚵🏽","").replace("🤾","").replace("🤾🏿","").replace("🤾🏻","").replace("🤾🏾","").replace("🤾🏼","").replace("🤾🏽","").replace("🤽","").replace("🤽🏿","").replace("🤽🏻","").replace("🤽🏾","").replace("🤽🏼","").replace("🤽🏽","").replace("🙎","").replace("🙎🏿","").replace("🙎🏻","").replace("🙎🏾","").replace("🙎🏼","").replace("🙎🏽","").replace("🙋","").replace("🙋🏿","").replace("🙋🏻","").replace("🙋🏾","").replace("🙋🏼","").replace("🙋🏽","").replace("🚣","").replace("🚣🏿","").replace("🚣🏻","").replace("🚣🏾","").replace("🚣🏼","").replace("🚣🏽","").replace("🏃","").replace("🏃🏿","").replace("🏃🏻","").replace("🏃🏾","").replace("🏃🏼","").replace("🏃🏽","").replace("🤷","").replace("🤷🏿","").replace("🤷🏻","").replace("🤷🏾","").replace("🤷🏼","").replace("🤷🏽","").replace("🏄","").replace("🏄🏿","").replace("🏄🏻","").replace("🏄🏾","").replace("🏄🏼","").replace("🏄🏽","").replace("🏊","").replace("🏊🏿","").replace("🏊🏻","").replace("🏊🏾","").replace("🏊🏼","").replace("🏊🏽","").replace("🛀","").replace("🛀🏿","").replace("🛀🏻","").replace("🛀🏾","").replace("🛀🏼","").replace("🛀🏽","").replace("💁","").replace("💁🏿","").replace("💁🏻","").replace("💁🏾","").replace("💁🏼","").replace("💁🏽","").replace("🚶","").replace("🚶🏿","").replace("🚶🏻","").replace("🚶🏾","").replace("🚶🏼","").replace("🚶🏽","").replace("👳","").replace("👳🏿","").replace("👳🏻","").replace("👳🏾","").replace("👳🏼","").replace("👳🏽","").replace("⛏","").replace("🥧","").replace("🐷","").replace("🐷","").replace("🐽","").replace("💩","").replace("💊","").replace("🎍","").replace("🍍","").replace("🏓","").replace("🔫","").replace("🍕","").replace("🛐","").replace("▶","").replace("⏯","").replace("🚓","").replace("🚨","").replace("👮","").replace("👮🏿","").replace("👮🏻","").replace("👮🏾","").replace("👮🏼","").replace("👮🏽","").replace("🐩","").replace("🎱","").replace("🍿","").replace("🏣","").replace("📯","").replace("📮","").replace("🍲","").replace("🚰","").replace("🥔","").replace("🍗","").replace("💷","").replace("😾","").replace("😡","").replace("📿","").replace("🤰","").replace("🤰🏿","").replace("🤰🏻","").replace("🤰🏾","").replace("🤰🏼","").replace("🤰🏽","").replace("🥨","").replace("🤴","").replace("🤴🏿","").replace("🤴🏻","").replace("🤴🏾","").replace("🤴🏼","").replace("🤴🏽","").replace("👸","").replace("👸🏿","").replace("👸🏻","").replace("👸🏾","").replace("👸🏼","").replace("👸🏽","").replace("🖨","").replace("🚫","").replace("💜","").replace("👛","").replace("📌","").replace("❓","").replace("🐰","").replace("🐰","").replace("🏎","").replace("📻","").replace("🔘","").replace("☢","").replace("🚃","").replace("🛤","").replace("🌈","").replace("🏳️‍🌈","").replace("🤚","").replace("🤚🏿","").replace("🤚🏻","").replace("🤚🏾","").replace("🤚🏼","").replace("🤚🏽","").replace("✊","").replace("✊🏿","").replace("✊🏻","").replace("✊🏾","").replace("✊🏼","").replace("✊🏽","").replace("✋","").replace("✋🏿","").replace("✋🏻","").replace("✋🏾","").replace("✋🏼","").replace("✋🏽","").replace("🙌","").replace("🙌🏿","").replace("🙌🏻","").replace("🙌🏾","").replace("🙌🏼","").replace("🙌🏽","").replace("🐏","").replace("🐀","").replace("⏺","").replace("♻","").replace("🍎","").replace("🔴","").replace("❤","").replace("🏮","").replace("🔻","").replace("🔺","").replace("®","").replace("😌","").replace("🎗","").replace("🔁","").replace("🔂","").replace("⛑","").replace("🚻","").replace("◀","").replace("💞","").replace("🦏","").replace("🎀","").replace("🍙","").replace("🍘","").replace("🤜","").replace("🤜🏿","").replace("🤜🏻","").replace("🤜🏾","").replace("🤜🏼","").replace("🤜🏽","").replace("🗯","").replace("➡","").replace("⤵","").replace("↩","").replace("⤴","").replace("💍","").replace("🍠","").replace("🤖","").replace("🚀","").replace("🗞","").replace("🎢","").replace("🤣","").replace("🐓","").replace("🌹","").replace("🏵","").replace("📍","").replace("🏉","").replace("🎽","").replace("👟","").replace("😥","").replace("⛵","").replace("🍶","").replace("🥪","").replace("📡","").replace("📡","").replace("🦕","").replace("🎷","").replace("🧣","").replace("🏫","").replace("🎒","").replace("✂","").replace("🦂","").replace("📜","").replace("💺","").replace("🙈","").replace("🌱","").replace("🤳","").replace("🤳🏿","").replace("🤳🏻","").replace("🤳🏾","").replace("🤳🏼","").replace("🤳🏽","").replace("🕢","").replace("🕖","").replace("🥘","").replace("☘","").replace("🦈","").replace("🍧","").replace("🌾","").replace("🛡","").replace("⛩","").replace("🚢","").replace("🌠","").replace("🛍","").replace("🛒","").replace("🍰","").replace("🚿","").replace("🦐","").replace("🔀","").replace("🤫","").replace("🤘","").replace("🤘🏿","").replace("🤘🏻","").replace("🤘🏾","").replace("🤘🏼","").replace("🤘🏽","").replace("🕡","").replace("🕕","").replace("⛷","").replace("🎿","").replace("💀","").replace("☠","").replace("🛷","").replace("😴","").replace("😪","").replace("🙁","").replace("🙂","").replace("🎰","").replace("🛩","").replace("🔹","").replace("🔸","").replace("😻","").replace("☺","").replace("😇","").replace("😍","").replace("😈","").replace("😊","").replace("😎","").replace("😏","").replace("🐌","").replace("🐍","").replace("🤧","").replace("🏔","").replace("🏂","").replace("🏂🏿","").replace("🏂🏻","").replace("🏂🏾","").replace("🏂🏼","").replace("🏂🏽","").replace("❄","").replace("☃","").replace("⛄","").replace("⚽","").replace("🧦","").replace("🍦","").replace("♠","").replace("🍝","").replace("❇","").replace("🎇","").replace("✨","").replace("💖","").replace("🙊","").replace("🔊","").replace("🔈","").replace("🔉","").replace("🗣","").replace("💬","").replace("🚤","").replace("🕷","").replace("🕸","").replace("🗓","").replace("🗒","").replace("🐚","").replace("🥄","").replace("🚙","").replace("🏅","").replace("🐳","").replace("🦑","").replace("😝","").replace("🏟","").replace("🤩","").replace("☪","").replace("✡","").replace("🚉","").replace("🍜","").replace("⏹","").replace("🛑","").replace("⏱","").replace("📏","").replace("🍓","").replace("🎙","").replace("🥙","").replace("☀","").replace("⛅","").replace("🌥","").replace("🌦","").replace("🌤","").replace("🌞","").replace("🌻","").replace("😎","").replace("🌅","").replace("🌄","").replace("🌇","").replace("🍣","").replace("🚟","").replace("💦","").replace("🕍","").replace("💉","").replace("👕","").replace("🌮","").replace("🥡","").replace("🎋","").replace("🍊","").replace("🚕","").replace("🍵","").replace("📆","").replace("☎","").replace("📞","").replace("🔭","").replace("📺","").replace("🕥","").replace("🕙","").replace("🎾","").replace("⛺","").replace("🌡","").replace("🤔","").replace("💭","").replace("🕞","").replace("🕒","").replace("👎","").replace("👎🏿","").replace("👎🏻","").replace("👎🏾","").replace("👎🏼","").replace("👎🏽","").replace("👍","").replace("👍🏿","").replace("👍🏻","").replace("👍🏾","").replace("👍🏼","").replace("👍🏽","").replace("🎫","").replace("🐯","").replace("🐯","").replace("⏲","").replace("😫","").replace("🚽","").replace("🍅","").replace("👅","").replace("🎩","").replace("🌪","").replace("🖲","").replace("🚜","").replace("™","").replace("🚋","").replace("🚊","").replace("🚋","").replace("🚩","").replace("📐","").replace("🔱","").replace("🚎","").replace("🏆","").replace("🍹","").replace("🐠","").replace("🎺","").replace("🌷","").replace("🥃","").replace("🦃","").replace("🐢","").replace("🕧","").replace("🕛","").replace("🐫","").replace("🕝","").replace("💕","").replace("👬","").replace("🕑","").replace("👭","").replace("☂","").replace("⛱","").replace("☔","").replace("😒","").replace("🦄","").replace("🔓","").replace("↕","").replace("↖","").replace("↗","").replace("⬆","").replace("🙃","").replace("🔼","").replace("🧛","").replace("🧛🏿","").replace("🧛🏻","").replace("🧛🏾","").replace("🧛🏼","").replace("🧛🏽","").replace("🚦","").replace("📳","").replace("✌","").replace("✌🏿","").replace("✌🏻","").replace("✌🏾","").replace("✌🏼","").replace("✌🏽","").replace("📹","").replace("🎮","").replace("📼","").replace("🎻","").replace("🌋","").replace("🏐","").replace("🖖","").replace("🖖🏿","").replace("🖖🏻","").replace("🖖🏾","").replace("🖖🏼","").replace("🖖🏽","").replace("🌘","").replace("🌖","").replace("⚠","").replace("🗑","").replace("⌚","").replace("🐃","").replace("🚾","").replace("🌊","").replace("🍉","").replace("👋","").replace("👋🏿","").replace("👋🏻","").replace("👋🏾","").replace("👋🏼","").replace("👋🏽","").replace("〰","").replace("🌒","").replace("🌔","").replace("🙀","").replace("😩","").replace("💒","").replace("🐳","").replace("☸","").replace("♿","").replace("⚪","").replace("❕","").replace("🏳","").replace("💮","").replace("✅","").replace("⬜","").replace("◽","").replace("◻","").replace("⭐","").replace("❔","").replace("▫","").replace("🔳","").replace("🥀","").replace("🎐","").replace("🌬","").replace("🍷","").replace("😉","").replace("😜","").replace("🐺","").replace("👩","").replace("👩‍🎨","").replace("👩🏿‍🎨","").replace("👩🏻‍🎨","").replace("👩🏾‍🎨","").replace("👩🏼‍🎨","").replace("👩🏽‍🎨","").replace("👩‍🚀","").replace("👩🏿‍🚀","").replace("👩🏻‍🚀","").replace("👩🏾‍🚀","").replace("👩🏼‍🚀","").replace("👩🏽‍🚀","").replace("🚴‍♀️","").replace("🚴🏿‍♀️","").replace("🚴🏻‍♀️","").replace("🚴🏾‍♀️","").replace("🚴🏼‍♀️","").replace("🚴🏽‍♀️","").replace("⛹️‍♀️","").replace("⛹🏿‍♀️","").replace("⛹🏻‍♀️","").replace("⛹🏾‍♀️","").replace("⛹🏼‍♀️","").replace("⛹🏽‍♀️","").replace("🙇‍♀️","").replace("🙇🏿‍♀️","").replace("🙇🏻‍♀️","").replace("🙇🏾‍♀️","").replace("🙇🏼‍♀️","").replace("🙇🏽‍♀️","").replace("🤸‍♀️","").replace("🤸🏿‍♀️","").replace("🤸🏻‍♀️","").replace("🤸🏾‍♀️","").replace("🤸🏼‍♀️","").replace("🤸🏽‍♀️","").replace("🧗‍♀️","").replace("🧗🏿‍♀️","").replace("🧗🏻‍♀️","").replace("🧗🏾‍♀️","").replace("🧗🏼‍♀️","").replace("🧗🏽‍♀️","").replace("👷‍♀️","").replace("👷🏿‍♀️","").replace("👷🏻‍♀️","").replace("👷🏾‍♀️","").replace("👷🏼‍♀️","").replace("👷🏽‍♀️","").replace("👩‍🍳","").replace("👩🏿‍🍳","").replace("👩🏻‍🍳","").replace("👩🏾‍🍳","").replace("👩🏼‍🍳","").replace("👩🏽‍🍳","").replace("💃","").replace("💃🏿","").replace("💃🏻","").replace("💃🏾","").replace("💃🏼","").replace("💃🏽","").replace("👩🏿","").replace("🕵️‍♀️","").replace("🕵🏿‍♀️","").replace("🕵🏻‍♀️","").replace("🕵🏾‍♀️","").replace("🕵🏼‍♀️","").replace("🕵🏽‍♀️","").replace("🧝‍♀️","").replace("🧝🏿‍♀️","").replace("🧝🏻‍♀️","").replace("🧝🏾‍♀️","").replace("🧝🏼‍♀️","").replace("🧝🏽‍♀️","").replace("🤦‍♀️","").replace("🤦🏿‍♀️","").replace("🤦🏻‍♀️","").replace("🤦🏾‍♀️","").replace("🤦🏼‍♀️","").replace("🤦🏽‍♀️","").replace("👩‍🏭","").replace("👩🏿‍🏭","").replace("👩🏻‍🏭","").replace("👩🏾‍🏭","").replace("👩🏼‍🏭","").replace("👩🏽‍🏭","").replace("🧚‍♀️","").replace("🧚🏿‍♀️","").replace("🧚🏻‍♀️","").replace("🧚🏾‍♀️","").replace("🧚🏼‍♀️","").replace("🧚🏽‍♀️","").replace("👩‍🌾","").replace("👩🏿‍🌾","").replace("👩🏻‍🌾","").replace("👩🏾‍🌾","").replace("👩🏼‍🌾","").replace("👩🏽‍🌾","").replace("👩‍🚒","").replace("👩🏿‍🚒","").replace("👩🏻‍🚒","").replace("👩🏾‍🚒","").replace("👩🏼‍🚒","").replace("👩🏽‍🚒","").replace("🙍‍♀️","").replace("🙍🏿‍♀️","").replace("🙍🏻‍♀️","").replace("🙍🏾‍♀️","").replace("🙍🏼‍♀️","").replace("🙍🏽‍♀️","").replace("🧞‍♀️","").replace("🙅‍♀️","").replace("🙅🏿‍♀️","").replace("🙅🏻‍♀️","").replace("🙅🏾‍♀️","").replace("🙅🏼‍♀️","").replace("🙅🏽‍♀️","").replace("🙆‍♀️","").replace("🙆🏿‍♀️","").replace("🙆🏻‍♀️","").replace("🙆🏾‍♀️","").replace("🙆🏼‍♀️","").replace("🙆🏽‍♀️","").replace("💇‍♀️","").replace("💇🏿‍♀️","").replace("💇🏻‍♀️","").replace("💇🏾‍♀️","").replace("💇🏼‍♀️","").replace("💇🏽‍♀️","").replace("💆‍♀️","").replace("💆🏿‍♀️","").replace("💆🏻‍♀️","").replace("💆🏾‍♀️","").replace("💆🏼‍♀️","").replace("💆🏽‍♀️","").replace("🏌️‍♀️","").replace("🏌🏿‍♀️","").replace("🏌🏻‍♀️","").replace("🏌🏾‍♀️","").replace("🏌🏼‍♀️","").replace("🏌🏽‍♀️","").replace("💂‍♀️","").replace("💂🏿‍♀️","").replace("💂🏻‍♀️","").replace("💂🏾‍♀️","").replace("💂🏼‍♀️","").replace("💂🏽‍♀️","").replace("👩‍⚕️","").replace("👩🏿‍⚕️","").replace("👩🏻‍⚕️","").replace("👩🏾‍⚕️","").replace("👩🏼‍⚕️","").replace("👩🏽‍⚕️","").replace("🧘‍♀️","").replace("🧘🏿‍♀️","").replace("🧘🏻‍♀️","").replace("🧘🏾‍♀️","").replace("🧘🏼‍♀️","").replace("🧘🏽‍♀️","").replace("🧖‍♀️","").replace("🧖🏿‍♀️","").replace("🧖🏻‍♀️","").replace("🧖🏾‍♀️","").replace("🧖🏼‍♀️","").replace("🧖🏽‍♀️","").replace("👩‍⚖️","").replace("👩🏿‍⚖️","").replace("👩🏻‍⚖️","").replace("👩🏾‍⚖️","").replace("👩🏼‍⚖️","").replace("👩🏽‍⚖️","").replace("🤹‍♀️","").replace("🤹🏿‍♀️","").replace("🤹🏻‍♀️","").replace("🤹🏾‍♀️","").replace("🤹🏼‍♀️","").replace("🤹🏽‍♀️","").replace("🏋️‍♀️","").replace("🏋🏿‍♀️","").replace("🏋🏻‍♀️","").replace("🏋🏾‍♀️","").replace("🏋🏼‍♀️","").replace("🏋🏽‍♀️","").replace("👩🏻","").replace("🧙‍♀️","").replace("🧙🏿‍♀️","").replace("🧙🏻‍♀️","").replace("🧙🏾‍♀️","").replace("🧙🏼‍♀️","").replace("🧙🏽‍♀️","").replace("👩‍🔧","").replace("👩🏿‍🔧","").replace("👩🏻‍🔧","").replace("👩🏾‍🔧","").replace("👩🏼‍🔧","").replace("👩🏽‍🔧","").replace("👩🏾","").replace("👩🏼","").replace("👩🏽","").replace("🚵‍♀️","").replace("🚵🏿‍♀️","").replace("🚵🏻‍♀️","").replace("🚵🏾‍♀️","").replace("🚵🏼‍♀️","").replace("🚵🏽‍♀️","").replace("👩‍💼","").replace("👩🏿‍💼","").replace("👩🏻‍💼","").replace("👩🏾‍💼","").replace("👩🏼‍💼","").replace("👩🏽‍💼","").replace("👩‍✈️","").replace("👩🏿‍✈️","").replace("👩🏻‍✈️","").replace("👩🏾‍✈️","").replace("👩🏼‍✈️","").replace("👩🏽‍✈️","").replace("🤾‍♀️","").replace("🤾🏿‍♀️","").replace("🤾🏻‍♀️","").replace("🤾🏾‍♀️","").replace("🤾🏼‍♀️","").replace("🤾🏽‍♀️","").replace("🤽‍♀️","").replace("🤽🏿‍♀️","").replace("🤽🏻‍♀️","").replace("🤽🏾‍♀️","").replace("🤽🏼‍♀️","").replace("🤽🏽‍♀️","").replace("👮‍♀️","").replace("👮🏿‍♀️","").replace("👮🏻‍♀️","").replace("👮🏾‍♀️","").replace("👮🏼‍♀️","").replace("👮🏽‍♀️","").replace("🙎‍♀️","").replace("🙎🏿‍♀️","").replace("🙎🏻‍♀️","").replace("🙎🏾‍♀️","").replace("🙎🏼‍♀️","").replace("🙎🏽‍♀️","").replace("🙋‍♀️","").replace("🙋🏿‍♀️","").replace("🙋🏻‍♀️","").replace("🙋🏾‍♀️","").replace("🙋🏼‍♀️","").replace("🙋🏽‍♀️","").replace("🚣‍♀️","").replace("🚣🏿‍♀️","").replace("🚣🏻‍♀️","").replace("🚣🏾‍♀️","").replace("🚣🏼‍♀️","").replace("🚣🏽‍♀️","").replace("🏃‍♀️","").replace("🏃🏿‍♀️","").replace("🏃🏻‍♀️","").replace("🏃🏾‍♀️","").replace("🏃🏼‍♀️","").replace("🏃🏽‍♀️","").replace("👩‍🔬","").replace("👩🏿‍🔬","").replace("👩🏻‍🔬","").replace("👩🏾‍🔬","").replace("👩🏼‍🔬","").replace("👩🏽‍🔬","").replace("🤷‍♀️","").replace("🤷🏿‍♀️","").replace("🤷🏻‍♀️","").replace("🤷🏾‍♀️","").replace("🤷🏼‍♀️","").replace("🤷🏽‍♀️","").replace("👩‍🎤","").replace("👩🏿‍🎤","").replace("👩🏻‍🎤","").replace("👩🏾‍🎤","").replace("👩🏼‍🎤","").replace("👩🏽‍🎤","").replace("👩‍🎓","").replace("👩🏿‍🎓","").replace("👩🏻‍🎓","").replace("👩🏾‍🎓","").replace("👩🏼‍🎓","").replace("👩🏽‍🎓","").replace("🏄‍♀️","").replace("🏄🏿‍♀️","").replace("🏄🏻‍♀️","").replace("🏄🏾‍♀️","").replace("🏄🏼‍♀️","").replace("🏄🏽‍♀️","").replace("🏊‍♀️","").replace("🏊🏿‍♀️","").replace("🏊🏻‍♀️","").replace("🏊🏾‍♀️","").replace("🏊🏼‍♀️","").replace("🏊🏽‍♀️","").replace("👩‍🏫","").replace("👩🏿‍🏫","").replace("👩🏻‍🏫","").replace("👩🏾‍🏫","").replace("👩🏼‍🏫","").replace("👩🏽‍🏫","").replace("👩‍💻","").replace("👩🏿‍💻","").replace("👩🏻‍💻","").replace("👩🏾‍💻","").replace("👩🏼‍💻","").replace("👩🏽‍💻","").replace("💁‍♀️","").replace("💁🏿‍♀️","").replace("💁🏻‍♀️","").replace("💁🏾‍♀️","").replace("💁🏼‍♀️","").replace("💁🏽‍♀️","").replace("🧛‍♀️","").replace("🧛🏿‍♀️","").replace("🧛🏻‍♀️","").replace("🧛🏾‍♀️","").replace("🧛🏼‍♀️","").replace("🧛🏽‍♀️","").replace("🚶‍♀️","").replace("🚶🏿‍♀️","").replace("🚶🏻‍♀️","").replace("🚶🏾‍♀️","").replace("🚶🏼‍♀️","").replace("🚶🏽‍♀️","").replace("👳‍♀️","").replace("👳🏿‍♀️","").replace("👳🏻‍♀️","").replace("👳🏾‍♀️","").replace("👳🏼‍♀️","").replace("👳🏽‍♀️","").replace("🧕","").replace("🧕🏿","").replace("🧕🏻","").replace("🧕🏾","").replace("🧕🏼","").replace("🧕🏽","").replace("🧟‍♀️","").replace("👢","").replace("👚","").replace("👒","").replace("👡","").replace("👯‍♀️","").replace("🤼‍♀️","").replace("🚺","").replace("🗺","").replace("😟","").replace("🎁","").replace("🔧","").replace("✍","").replace("✍🏿","").replace("✍🏻","").replace("✍🏾","").replace("✍🏼","").replace("✍🏽","").replace("💛","").replace("💴","").replace("☯","").replace("🤪","").replace("🦓","").replace("🤐","").replace("🧟","").replace("💤","").replace("🇦🇽","")
        #    text1=meta
        #    print text1
        #     text2=split(text1,2000)
        text2=re.split('[!"#$%\()*+,-../:;<=>?@[\\]^_`{|}~]',re.sub('\S+@\S+|@\S+','',text1).strip())
#        text2=text1
        text=[]
        text3=text4=text5=regexdf=""
        
        for k in range(len(text2)):
            if len(text2[k])>4000:
                text3=split(text2,4000)
                for l in range(len(text3)):
                    regexdf=""
                    if len(text3[l]) > 2:
                        temptext=""
                        text4=""
                        temptxt=text3[l]
        #                text4=re.sub('[^A-Za-z0-9 ]+', ' ', text3[l])
                        if len(temptxt.strip()) !=0:
                            regexdf=' '.join(temptxt.split()).lower()
        #                         if l!=0: 
                            try:
                                username_box.send_keys(".".decode('utf8'))
                                clearbox.click()
            #                         print regexdf
                                username_box.send_keys(regexdf.strip().decode('utf8'))
                                sleep(0.5)
                                username_box.send_keys(".".decode('utf8'))
                             
#                                sleep(1)
#                                transbox=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div/div/div[2]/div[2]/div/div/div')
#                                transbox.click()
#                                counter+=1

                                sleep(random.randint(3,4))
                            


                                trans_text=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[3]/div[1]/div[2]/div')
                                text5=trans_text.text
#                                pyperclip.copy("")
                                sleep(random.randint(2,3))
                                text.append(text5)
                                
                            except Exception as e:
                                pass
                        text.append(text4)
                        
                        
                        
            else:
                regexdf=""
                if len(text2[k]) > 2:
                    temptext=""
                    text4=""
                    temptxt=text2[k]
        #            text4=re.sub('[^A-Za-z0-9 ]+', ' ', text2[k])
                    if len(temptxt.strip()) !=0:
                        regexdf=' '.join(temptxt.split()).lower()
        #                     if k!=0:    
                        try:
                            username_box.send_keys(".".decode('utf8'))
                            clearbox.click()
            #                     print regexdf
                            username_box.send_keys(regexdf.strip().decode('utf8'))
                            sleep(0.5)
                            username_box.send_keys(".".decode('utf8'))
                         
#                            sleep(1)
#                            transbox=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div/div/div[2]/div[2]/div/div/div')
#                            transbox.click()
#                            counter+=1

                            sleep(random.randint(3,4))

                            trans_text=driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[3]/div[1]/div[2]/div')
                            text5=trans_text.text
#                            pyperclip.copy("")
                            sleep(random.randint(2,3))
                            text.append(text5)
                            
                        except Exception as e:
                            pass
                    text.append(text4)
                      
           
            trans=" ".join(text)
        #        if counter>100:
        #            counter=0
        #            sleep(120)
        
        #    print trans
        tmptxt1=" ".join(trans.split())
        file3.iloc[i,2]=tmptxt1
        print i
        print file3.iloc[i,1]
        print file3.iloc[i,2]
    return file3




# =============================================================================
# Reading Video Ids
# =============================================================================
#translatescrp.translatescrp
def testid(resp):
    
    testids=pd.read_csv("test_vid_ids.csv",header=None)
    
    #Fetching meta
    file1 = pd.DataFrame(index=range(len(testids.index)), columns=["id","Meta"])
    file1 = file1.fillna("")
    
    for i in range(len(testids)):
        file1.iloc[i,0]=testids.iloc[i,0]
        file1.iloc[i,1]=Cleaning.vidmeta(testids.iloc[i,0])
    #    if len(file3.iloc[i,1])!=0:
    ##        file3.iloc[i,2]=Cleaning.singletxt(file3.iloc[i,1])
    ##        file3.iloc[i,2]=Cleaning.singletxtfrn(file3.iloc[i,1],"auto")
    #        file3.iloc[i,2]=translatescrp.translatescrp(file3.iloc[i,1])
    #        file3.iloc[i,2]=Cleaning.translatepkg(file3.iloc[i,1])
    #    file3.iloc[i,3]=df.iloc[i,2]
        print i
    #    print file3.iloc[i,1]
    #    print file3.iloc[i,2]
    file1.to_csv("vidid_Raw_meta.csv")
    
    if resp==1:
        file2=translatescrp1(file1)
    elif resp==2:
        file2=translatescrp2(file1)
    
    file2.to_csv("vidid_transmeta.csv",encoding="utf-8")
    
    df1=translatescrp.clean(file2)
    
    file3 = df1[df1.CleanMeta!='']
    file3.index=range(len(file3.index))
    
    
    #file3 = file3.drop_duplicates('Meta')
    file3['Meta'].replace('', np.nan, inplace=True)
    file3.dropna(subset=['Meta'], inplace=True)
    
    file3.to_csv("vidid_trans&cleanmeta.csv",encoding="utf-8")
    
    print "Cleaned and exported"
    X3_test = loaded_tfidf.transform(file3.CleanMeta).toarray()
    X3_test.shape
    y3_pred=loaded_model.predict(X3_test)
    
    file3['pred_cat_id'] = y3_pred
    s2=[]
    for i in range(len(y3_pred)):
    #     s1.append(id_to_category[y1_pred[i]])
        s2.append(loaded_id_to_category[y3_pred[i]])
    file3['pred_sub'] = s2
    
    #Fine Tuning Model
#    filename4="Min_thresh_5.sav"
#    min_thresh = pickle.load(open(filename4, 'rb'))
    fin_s=[]
    confidence=[]
    definition=[]
    for i in range(len(X3_test)):
        b=loaded_model.predict_proba(X3_test[i].reshape(1, -1))*100
        #b=model1.decision_function(X2_test[i].reshape(1, -1))
        if round(np.amax(b))<round(min_thresh[loaded_id_to_category[np.argmax(b)]]):
            categ="NA"
        elif round(np.amax(b))<30:
            categ="NA"
#            fin_s.append("NA")
        else:
            categ=loaded_id_to_category[np.argmax(b)]
#            fin_s.append(loaded_id_to_category[np.argmax(b)])
        fin_s.append(categ)
        confidence.append(str(round(np.max(b),1))+"%")
        definition.append(desc(categ))
        
    file3['fin_pred'] = fin_s
    file3['Confidence'] = confidence
    file3['Definition'] = definition
    return file3

# =============================================================================
# For Channels
# =============================================================================

def testchid(chids,resp):
    file1=translatescrp.channelvid(chids,50,1)
    
    file1=file1.iloc[:,0:2]
    
    file3 = pd.DataFrame(index=range(len(file1.index)), columns=["id","Meta","CleanMeta"])
    file3 = file3.fillna("")
    
        

    if resp==1:
        file2=translatescrp1(file1)
    elif resp==2:
        file2=translatescrp2(file1)

        

            
    df1=translatescrp.clean(file2)
    
    file3 = df1[df1.CleanMeta!='']
    file3.index=range(len(file3.index))
    
    
    #file3 = file3.drop_duplicates('Meta')
    file3['Meta'].replace('', np.nan, inplace=True)
    file3.dropna(subset=['Meta'], inplace=True)
    
    
    X3_test = loaded_tfidf.transform(file3.CleanMeta).toarray()
    X3_test.shape
    y3_pred=loaded_model.predict(X3_test)
    
    file3['pred_cat_id'] = y3_pred
    s2=[]
    for i in range(len(y3_pred)):
    #     s1.append(id_to_category[y1_pred[i]])
        s2.append(loaded_id_to_category[y3_pred[i]])
    file3['pred_sub'] = s2
    
    #Fine Tuning Model
#    filename4="Min_thresh_5.sav"
#    min_thresh = pickle.load(open(filename4, 'rb'))
    fin_s=[]
    confidence=[]
    definition=[]
    for i in range(len(X3_test)):
        b=loaded_model.predict_proba(X3_test[i].reshape(1, -1))*100
        #b=model1.decision_function(X2_test[i].reshape(1, -1))
        if round(np.amax(b))<round(min_thresh[loaded_id_to_category[np.argmax(b)]]):
            categ="NA"
        elif round(np.amax(b))<30:
            categ="NA"
#            fin_s.append("NA")
        else:
            categ=loaded_id_to_category[np.argmax(b)]
#            fin_s.append(loaded_id_to_category[np.argmax(b)])
        fin_s.append(categ)
        confidence.append(str(round(np.max(b),1))+"%")
        definition.append(desc(categ))
    
    file3['fin_pred'] = fin_s
    file3['Confidence'] = confidence
    file3['Definition'] = definition
    

#    b = {}
#    for item in file3['fin_pred']:
#        b[item] = b.get(item, 0) + 1
    

    return file3

# =============================================================================

#profile = FirefoxProfile("C:\Users\Administrator\AppData\Roaming\Mozilla\Firefox\Profiles\uw67m271.default-release")
#profile.set_preference('devtools.jsonview.enabled', False)
#driver = webdriver.Firefox(firefox_profile=profile,executable_path=r"C:\Server backup\VIDOOLY NILANJAN\Python\Selenium Drivers\geckodriver-v0.20.1-win64\geckodriver.exe")

# =============================================================================
def testchid_vect2(chids):
#    chids="UC_wJC5fCX-z61E3htXxQvKQ"
    driver.get("http://localhost:5013/RnD/trans_try?id="+chids)
    sleep(2)
    
    select = Select(driver.find_element_by_class_name("goog-te-combo"))
    
    select.select_by_visible_text("English")
    
    sleep(8)
#    text=[]
    txt1=driver.find_elements_by_class_name("text")[0].text.split("\n")
    
    file1=translatescrp.channelvid(chids,50,1)
    
    video_id=file1['id'].tolist()
    Translated=file1['meta'].tolist()
    
    data1=pd.DataFrame({"video_id":video_id,"Translated":Translated,"translated_meta":txt1})

    cols=data1.columns.tolist()
    cols=[u'video_id', 'Translated', u'translated_meta']
    data1=data1[cols]
    
    df1=translatescrp.clean(data1)
    
    file3 = df1[df1.CleanMeta!='']
    file3.index=range(len(file3.index))
    
    
    #file3 = file3.drop_duplicates('Meta')
    file3['Meta'].replace('', np.nan, inplace=True)
    file3.dropna(subset=['Meta'], inplace=True)
    
    
    X3_test = loaded_tfidf.transform(file3.CleanMeta).toarray()
#    X3_test.shape
    
    fin_s=[]
    confidence=[]
#    definition=[]
    for i in range(len(X3_test)):
        b=loaded_model.predict_proba(X3_test[i].reshape(1, -1))*100
        #b=model1.decision_function(X2_test[i].reshape(1, -1))
        if round(np.amax(b))<round(min_thresh[loaded_id_to_category[np.argmax(b)]]):
            fin_s.append("NA")
        elif round(np.amax(b))<40:
            fin_s.append("NA")
        else:
            fin_s.append(loaded_id_to_category[np.argmax(b)])
#        confidence.append(np.max(b))
        confidence.append(str(round(np.max(b),1))+"%")
#        definition.append(desc(categ))
    
    file3['fin_pred'] = fin_s
    file3['Confidence'] = confidence
#    file3['Definition'] = definition
    
    return file3

# =============================================================================
# Testing Video Ids
# =============================================================================
file3=testid(1)
print file3['fin_pred']
file3.columns.tolist()
file4=file3[['id','CleanMeta','fin_pred','Confidence','Definition']]

file4.to_csv('Video_cat.csv',sep="\t",encoding="utf-8")


# =============================================================================
# Testing Channel ids
# =============================================================================
chids=pd.read_csv("test_ch_ids.csv",header=None)

file4 = pd.DataFrame(index=range(0), columns=['Channel_Id','id','Meta','Transmeta','CleanMeta','pred_cat_id','pred_sub','fin_pred','Confidence','Definition'])

for i in range(len(chids)):
    chid=chids.iloc[i,0]
    file3=testchid(chid,1)
    file3['Channel_Id']=chid
    file4=file4.append(file3)
    print chid
    

# =============================================================================
# Testing Channel ids Vect 2
# =============================================================================
# =============================================================================

#profile = FirefoxProfile("C:\Users\Administrator\AppData\Roaming\Mozilla\Firefox\Profiles\uw67m271.default-release")
##profile.set_preference('devtools.jsonview.enabled', False)
#driver = webdriver.Firefox(firefox_profile=profile,executable_path=r"C:\Server backup\VIDOOLY NILANJAN\Python\Selenium Drivers\geckodriver-v0.20.1-win64\geckodriver.exe")

driver = webdriver.Chrome(r'C:\Server backup\VIDOOLY NILANJAN\Python\Selenium Drivers\chromedriver_win32\chromedriver.exe') 
#driver.minimize_window()

sleep(5)
# =============================================================================
chids=pd.read_csv("test_ch_ids.csv",header=None)
len(chids)
file4 = pd.DataFrame(index=range(0), columns=['Channel_Id','id','Meta','Transmeta','CleanMeta','fin_pred','Confidence'])
n=1
for i in range(len(chids)):
#    i=1
    start=time.time()
    chid=chids.iloc[i,0]
    try:
        file3=testchid_vect2(chid)
    except Exception as e:
        continue
    file3['Channel_Id']=chid
    file4=file4.append(file3)
    if n>20:
        file_temp=file4[['Channel_Id','id','Meta','Transmeta','CleanMeta','fin_pred','Confidence']]
        file_temp.to_csv('Channel_cat_temp3.csv' ,encoding="utf-8", sep="\t")
        n=1
    n+=1
    print i,chid,n
    end=time.time()
    print end-start
    

file5=file4[['Channel_Id','id','Meta','Transmeta','CleanMeta','fin_pred','Confidence']]

file5.to_csv('Channel_cat3.csv' ,encoding="utf-8", sep="\t")  

