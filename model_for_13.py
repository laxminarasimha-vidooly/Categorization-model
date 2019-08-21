#Importing libraries
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import numpy as np
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
#from googletrans import Translator
#translator = Translator()
import string
import urllib
import requests
import json
import nltk
import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#nltk.download('stopwords')
import os,sys
import Cleaning
import translatescrp
import pickle
os.getcwd()

#Text preprocessing
def clean(transdf):
    regexdf = pd.DataFrame( index=range(len(transdf.index)), columns=["id","Translated","Clean_Translated","sub"])
    regexdf = regexdf.fillna("")
    
    for i in range(len(transdf)):
        regexdf.iloc[i,0]=transdf.iloc[i,0]
        regexdf.iloc[i,1]=transdf.iloc[i,2]
        
        
        temptxt=re.sub('[^A-Za-z0-9 ]+', ' ', transdf.iloc[i,2])
        regexdf.iloc[i,2]=' '.join(temptxt.split()).lower()
        
        regexdf.iloc[i,3]=transdf.iloc[i,3]
        print i
    print "1st cleaning done"
        
#    regexdf.to_csv('D:/Working/Auto-sub cat modelling/SVM/SVM/clean_translated.csv', encoding='utf-8')
    
    # =============================================================================
    # ###Reading Stopwords###
    # =============================================================================
    stop_words=pd.read_csv('E:/Laxmi_Rnd/My Laptop/Model for cat/stop_words.csv',encoding='utf-8',header=None)
    stop_word=[]
    for i in range(len(stop_words)):
        stop_word.append(stop_words.iloc[i,0])
        
    stpdf = pd.DataFrame(index=range(len(regexdf.index)), columns=["id","Clean_Translated","stopword_cleaned","sub"])
    stpdf = stpdf.fillna("")
    
    # =============================================================================
    # ### Removing stopwords###
    # =============================================================================
    for i in range(len(regexdf.index)):
        stpdf.iloc[i,0]=regexdf.iloc[i,0]
        stpdf.iloc[i,1]=regexdf.iloc[i,2]
        
        words = regexdf.iloc[i,2].split()
        str_list = []
        for r in words:
            if not r in stop_word:
                str_list.append(r)
        stpdf.iloc[i,2]=" ".join(str_list)
        
        stpdf.iloc[i,3]=regexdf.iloc[i,3]
        print i
    print "stop words cleaning done"
    
#    stpdf.to_csv('D:/Working/Auto-sub cat modelling/SVM/SVM/stopword_cleaned.csv', encoding='utf-8')
    
    # =============================================================================
    # ##stemming
    # =============================================================================
    
    ps = PorterStemmer()
    #nltk.download('punkt')
    stmdf = pd.DataFrame(index=range(len(stpdf.index)), columns=["id","stopword_cleaned","stem_cleaned","sub"])
    stmdf = stmdf.fillna("")
    
    for i in range(len(stpdf.index)):
        stmdf.iloc[i,0]=stpdf.iloc[i,0]
        stmdf.iloc[i,1]=stpdf.iloc[i,2]
        words = word_tokenize(stpdf.iloc[i,2].decode('utf8'))
        wrd=[]
        for w in words:
            wrd.append(ps.stem(w).encode('utf8'))
        stmdf.iloc[i,2]=" ".join(wrd)
        stmdf.iloc[i,3]=stpdf.iloc[i,3]
        print i
    print "stemming done"
#    stmdf.to_csv('D:/Working/Auto-sub cat modelling/SVM/SVM/stem_cleaned.csv', encoding='utf-8')
    
    fnldf = pd.DataFrame(index=range(len(stmdf.index)), columns=["id","stem_cleaned","final_clean","sub"])
    fnldf = fnldf.fillna("")
    
    ### Final Cleaning###
    for i in range(len(stmdf.index)):
        fnldf.iloc[i,0]=stmdf.iloc[i,0]
        fnldf.iloc[i,1]=stmdf.iloc[i,2]
        
        words = stmdf.iloc[i,2].split()
        str_list = []
        for r in words:
            if len(r)>1:        
                str_list.append(r)
            else:
                if len(re.sub('[0-9 ]+', '', r))==0:
                    str_list.append(r)
        fnldf.iloc[i,2]=" ".join(str_list)
        
        fnldf.iloc[i,3]=stmdf.iloc[i,3]
        print i
    print "Final cleaning done"
        
#    fnldf.to_csv('D:/Working/Auto-sub cat modelling/SVM/SVM/final_clean.csv', encoding='utf-8')
    return fnldf    

#Reading raw text (training)
data1=pd.read_csv("new_sampled_data.csv",encoding='utf-8')
data1.head()
data1=data1.iloc[:,1:4]
data1['category'].unique()
data1.iloc[0,:]
data1['Translated']='NA'
cols=data1.columns.tolist()
cols=[u'id', 'Translated', u'Transmeta', u'category']

data1=data1[cols]

data2=clean(data1)
data2.to_csv('final_clean_5.csv', encoding='utf-8')

df=pd.read_csv("final_clean_5.csv",encoding='utf-8')
df.head()
df=df.iloc[:,1:5]
df['sub'].unique()


# =============================================================================
# Train Test Split and Export
# =============================================================================

train = pd.DataFrame(index=range(0), columns=["id","stem_cleaned","final_clean","sub"])
train = train.fillna("")
test = pd.DataFrame(index=range(0), columns=["id","stem_cleaned","final_clean","sub"])
test = test.fillna("")

for subc in list(df['sub'].unique()):
    tmpdf=df[(df['sub'] == subc)]
    tr, te = train_test_split(tmpdf, test_size=0.2)
    train = train.append(tr)
    test = test.append(te)
    
    
train['sub'].value_counts()
test['sub'].value_counts()
#train.head()

train.to_csv("train.csv", sep=',', encoding='utf-8')
test.to_csv("test.csv", sep=',', encoding='utf-8')    

# =============================================================================
# Importing Train
# =============================================================================

fnldf=pd.read_csv("train.csv")

np.where(pd.isnull(fnldf))


len(fnldf)
fnldf.head()
fnldf=fnldf.iloc[:,1:5]
# =============================================================================
# Assigning category_id
# =============================================================================
fnldf['category_id'] = fnldf['sub'].factorize()[0]
category_id_df = fnldf[['sub', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'sub']].values)
fnldf.tail()
# =============================================================================
# Checking balance
# =============================================================================
fig = plt.figure(figsize=(8,6))
fnldf.groupby('sub').id.count().plot.bar(ylim=0)
plt.show()


#############
#repeat=range(11,22)
#for value in repeat:
# =============================================================================
# Calling TfidfVectorizer and assigning features(X) and lebels(Y)
# =============================================================================
#    while True:
#        try:
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,norm='l2', encoding='latin-1', ngram_range=(1, 2))
features = tfidf.fit_transform(fnldf.final_clean).toarray()

labels = fnldf.category_id
features.shape
# =============================================================================
# Model Creation
# =============================================================================

#model1 = svm.SVC(kernel='linear')
# X1_train, X1_test, y1_train, y1_test, indices_train, indices_test = train_test_split(features, labels, fnldf.index, test_size=0.33, random_state=0)
X2_train=features
y2_train=labels.astype('int')
# y1_test=y1_test.astype('int')

model1=svm.SVC(kernel='linear', probability=True)
model2=model1.fit(X2_train, y2_train)
#            print value
#        except:
#            value=value+1
#            print "value is",value
#            continue



# save the model to disk
filename = 'finalized_model_final.sav'
joblib.dump(model2, open(filename, 'wb'))
filename1 = 'tfidf_final.sav'
joblib.dump(tfidf, open(filename1, 'wb'))
filename2 = 'category_to_id_final.sav'
joblib.dump(category_to_id, open(filename2, 'wb'))
filename3 = 'id_to_category_final.sav'
joblib.dump(id_to_category, open(filename3, 'wb'))

# =============================================================================
# Importing Clean Test
# =============================================================================

testdf=pd.read_csv("test.csv")
len(testdf)
testdf.head()
testdf=testdf.iloc[:,1:5]

# =============================================================================
# Calling TfidfVectorizer and assigning features(X) to test
# =============================================================================

X2_test = tfidf.transform(testdf.final_clean).toarray()
X2_test.shape
# =============================================================================
# prediction
# =============================================================================
y2_pred=model2.predict(X2_test)

# =============================================================================
# Reading Video Ids
# =============================================================================
#translatescrp.translatescrp
def testid(resp):
    
    testids=pd.read_csv("testids.csv")
    
    #Fetching meta
    file1 = pd.DataFrame(index=range(len(testids.index)), columns=["id","Meta"])
    file1 = file1.fillna("")
    
    for i in range(len(testids)):
        file1.iloc[i,0]=testids.iloc[i,0]
        file1.iloc[i,1]=Cleaning.vidmeta(testids.iloc[i,0])

        
    if resp==1:
        file2=translatescrp.translatescrp(file1)
    elif resp==2:
        file2=translatescrp.translatescrpfrn(file1)
    
    
    df1=translatescrp.clean(file2)
    
    file3 = df1[df1.CleanMeta!='']
    file3.index=range(len(file3.index))
    
    
    #file3 = file3.drop_duplicates('Meta')
    file3['Meta'].replace('', np.nan, inplace=True)
    file3.dropna(subset=['Meta'], inplace=True)
    
    X3_test = tfidf.transform(file3.CleanMeta).toarray()
    X3_test.shape
    y3_pred=model2.predict(X3_test)
    
    file3['pred_cat_id'] = y3_pred
    s2=[]
    for i in range(len(y3_pred)):
    #     s1.append(id_to_category[y1_pred[i]])
        s2.append(id_to_category[y3_pred[i]])
    file3['pred_sub'] = s2
    
    #Fine Tuning Model
    filename4="Min_thresh_5.sav"
    min_thresh = pickle.load(open(filename4, 'rb'))
    fin_s=[]
    for i in range(len(X3_test)):
        b=model2.predict_proba(X3_test[i].reshape(1, -1))*100
        #b=model1.decision_function(X2_test[i].reshape(1, -1))
        if round(np.amax(b))<round(min_thresh[id_to_category[np.argmax(b)]]):
            fin_s.append("NA")
        else:
            fin_s.append(id_to_category[np.argmax(b)])
    
    file3['fin_pred'] = fin_s
    return file3
    
#Calling prediction function along with translation parameter
file3=testid(1)
print file3['fin_pred']  

#file3.to_csv("C:/Users/dell/Desktop/VIDOOLY/NEW/Python/Naive Bais/Updated Code/TRY2/test_pred_5.csv", sep='\t', encoding='utf-8')
#file3.iloc[9,3]

# =============================================================================
# assigning actual cat id
# =============================================================================
s1=[]
for i in range(len(testdf)):
#     s1.append(id_to_category[y1_pred[i]])
    s1.append(category_to_id[testdf.iloc[i,3]])
testdf['actal_cat_id'] = s1
testdf.head()
# =============================================================================
# assigning predicted cat id
# =============================================================================
testdf['pred_cat_id'] = y2_pred
# =============================================================================
# assigning predicted categories
# =============================================================================
s2=[]
for i in range(len(y2_pred)):
#     s1.append(id_to_category[y1_pred[i]])
    s2.append(id_to_category[y2_pred[i]])
testdf['pred_sub'] = s2
testdf.head()
# =============================================================================
# exporting predicted test
# =============================================================================
testdf.to_csv("test_pred_subcat.csv", sep='\t', encoding='utf-8')

###############################################################################
#model confidence measure
###############################################################################
b=model1.decision_function(X2_test)
#b=loaded_model.decision_function(X3_test)

a=model1.predict_proba(X2_test)*100
#a=loaded_model.predict_proba(X3_test)*100
print ("max percentage for classifier is", round(np.amax(a)))
print ("max of decision function is", round(np.amax(b)))

#Threshold= 40.0
#type(float(np.amax(b)))
# Filtering Test data with correct Prediction
success_count=testdf[(testdf['actal_cat_id']==testdf['pred_cat_id'])]
len(success_count)
len(testdf)
success_count.head
#Changing it to TFIDF
X2_success = tfidf.transform(success_count.final_clean).toarray()
X2_success.shape

#Getting maximum Probability values for every Category in a dictionary
prob=dict()
for i in range(len(success_count)):
    key=success_count.iloc[i,3]
    prob.setdefault(key,[]).append(np.amax(model1.predict_proba(X2_success[i].reshape(1, -1))*100))
    #prob.setdefault(key,[]).append(np.amax(model1.decision_function(X2_success[i].reshape(1, -1))))
len(prob.keys())

#Getting Min values of all max values as threshold
thresh=dict()
for key, value in prob.iteritems():
    thresh[key]= min(value)
    
thresh_max=dict()
for key, value in prob.iteritems():
    thresh_max[key]= max(value)
    
#Exporting minimum Threshold Values
filename4 = 'Min_thresh_final.sav'
joblib.dump(thresh, open(filename4, 'wb'))

fins=[]
confidence=[]
for i in range(len(X2_test)):
    b=model1.predict_proba((X2_test[1].reshape(1, -1)))*100
    #b=model1.predict_proba((X2_test[2].reshape(1, -1)))*100
    #b=model1.decision_function(X2_test[i].reshape(1, -1))
    if round(np.amax(b))<round(thresh[id_to_category[np.argmax(b)]]):
        fins.append("NA")
    else:
        fins.append(id_to_category[np.argmax(b)])
    confidence.append(np.max(b))
testdf['fin_pred'] = fins
testdf['confidence'] = confidence

# =============================================================================
# confusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
conf_mat = confusion_matrix(testdf.iloc[:,4].values, testdf.iloc[:,5].values)
df_cm = pd.DataFrame(conf_mat,id_to_category.values(),id_to_category.values())                  
fig, ax = plt.subplots(figsize=(20,20))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.0)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 11})# font size
plt.ylabel('actual')
plt.xlabel('predicted')
plt.show()

# =============================================================================
#Model Accuracy
# =============================================================================

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(testdf.iloc[:,4].values, testdf.iloc[:,5].values)
# =============================================================================
# model accuracy
# =============================================================================
train_score=model2.score (X2_train, y2_train)
# =============================================================================
# print test accuracy and training accuracy
# =============================================================================
print("Test accuracy is", accuracy, "trainning accuracy is",train_score)
