# Gender-Identification-from-Text
An application, which identifies author's gender from short text (articles/blogs) by extracting features using Python Scikit (Scipy toolkit) and training these data features with the developed predictive model using machine learning algorithms.

#Procedure to execute the code snippet files included along with this file:
1) Restore the database from a sql backup file named - 'Gender_blog_dataset.bak'(provided in the folder) using SQL SERVER 2012
	A sample database is created in sql with table names 'test', 'train', 'dataset'
2) Install numpy, nltk, re, sklearn libraries in python using 'pip install'
3) Open Python ILDE
4) Execute GenderClassification_All_Features.py
5) Execute Initial_features.py
6) Execute No_POS_Features.py
