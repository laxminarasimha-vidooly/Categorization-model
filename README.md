# Model for categorizing videos based on Text

Project:

Classification of videos based on video title + description (1st para)

Problem statement:

Videos needs to be categorized according to pre-defined classes (as decided by the insights requirement) as the classification of videos helps in providing efficient custom reports such as genre insights in audience engagement.

Tools used: Python, selenium web drivers

Technique used: SVM

Process:

Training:
1.	34 broad categories were defined as per insights requirement
2.	Approximately 1.5k video were labelled based on text
3.  Text translation, preprocessing (removal of punctuations, white space, stop words, emojis, special characters, stemming etc.,) and splitting data into train and test 
3.	Trained model using SVM (linear kernel) with 80% of whole data

Testing:
1.	Collection of video text (title + description) using YT API
2.	Conversion of Non – English text to English text using google translation (selenium)
3.  Text pre-processing (removal of punctuations, white space, stop words, emojis, special characters, stemming etc.,)
4.	Final cleaning of text for removing untranslated text (emojis and other langauges which are not in google translate)
5.	Prediction using model and testing accuracy

Accuracy:
Model showed the accuracy of 89% on test data. 

