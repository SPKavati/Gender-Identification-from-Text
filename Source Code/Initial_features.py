from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import preprocessing
import pymssql
import numpy as num
import nltk
from collections import Counter
print "started"

#Connection to sql
conn = pymssql.connect(server='localhost:1433', user='sa', password='sqlserver', database='sample')
print "success"
cursor = conn.cursor()

#import train data from SQL server
cursor.execute('SELECT * from train;')
train_features=[]
doc=[]
gender=[]
pos_train=[]

REGEX = [ ';','!','@','#','&',':)',':(',':P',':D',';-)' ]   #';','{','}',':','"','?',
stop="a, about, above, across, after, again, against, all, almost, alone, along, already, also, although, always, am, among, an, and, another, any, anybody, anyone, anything, anywhere, are, area, areas, aren't, around, as, ask, asked, asking, asks, at, away, b, back, backed, backing, backs, be, became, because, become, becomes, been, before, began, behind, being, beings, below, best, better, between, big, both, but, by, c, came, can, cannot, can't, case, cases, certain, certainly, clear, clearly, come, could, couldn't, d, did, didn't, differ, different, differently, do, does, doesn't, doing, done, don't, down, downed, downing, downs, during, e, each, early, either, end, ended, ending, ends, enough, even, evenly, ever, every, everybody, everyone, everything, everywhere, f, face, faces, fact, facts, far, felt, few, find, finds, first, for, four, from, full, fully, further, furthered, furthering, furthers, g, gave, general, generally, get, gets, give, given, gives, go, going, good, goods, got, great, greater, greatest, group, grouped, grouping, groups, h, had, hadn't, has, hasn't, have, haven't, having, he, he'd, he'll, her, here, here's, hers, herself, he's, high, higher, highest, him, himself, his, how, however, how's, i, i'd, if, i'll, i'm, important, in, interest, interested, interesting, interests, into, is, isn't, it, its, it's, itself, i've, j, just, k, keep, keeps, kind, knew, know, known, knows, l, large, largely, last, later, latest, least, less, let, lets, let's, like, likely, long, longer, longest, m, made, make, making, man, many, may, me, member, members, men, might, more, most, mostly, mr, mrs, much, must, mustn't, my, myself, n, necessary, need, needed, needing, needs, never, new, newer, newest, next, no, nobody, non, noone, nor, not, nothing, now, nowhere, number, numbers, o, of, off, often, old, older, oldest, on, once, one, only, open, opened, opening, opens, or, order, ordered, ordering, orders, other, others, ought, our, ours, ourselves, out, over, own, p, part, parted, parting, parts, per, perhaps, place, places, point, pointed, pointing, points, possible, present, presented, presenting, presents, problem, problems, put, puts, q, quite, r, rather, really, right, room, rooms, s, said, same, saw, say, says, second, seconds, see, seem, seemed, seeming, seems, sees, several, shall, shan't, she, she'd, she'll, she's, should, shouldn't, show, showed, showing, shows, side, sides, since, small, smaller, smallest, so, some, somebody, someone, something, somewhere, state, states, still, such, sure, t, take, taken, than, that, that's, the, their, theirs, them, themselves, then, there, therefore, there's, these, they, they'd, they'll, they're, they've, thing, things, think, thinks, this, those, though, thought, thoughts, three, through, thus, to, today, together, too, took, toward, turn, turned, turning, turns, two, u, under, until, up, upon, us, use, used, uses, v, very, w, want, wanted, wanting, wants, was, wasn't, way, ways, we, we'd, well, we'll, wells, went, were, we're, weren't, we've, what, what's, when, when's, where, where's, whether, which, while, who, whole, whom, who's, whose, why, why's, will, with, within, without, won't, work, worked, working, works, would, wouldn't, x, y, year, years, yes, yet, you, you'd, you'll, young, younger, youngest, your, you're, yours, yourself, yourselves, you've, z"
PROCESS=[';','!','@','#','&',':)',':(',':P',':D',';-)','-','\'','.',',','{','}',':','"','?',]
#Loop through each review
for row in cursor:
    feature_list=[]
    distinct={}
    pos_list=[]
    rx=re.compile(r'(.)\1{5,}')
    sentence=row[0].replace("&nbsp","")
    text = nltk.tokenize.word_tokenize(sentence)
    bigrams=nltk.bigrams(text)
    fdist = nltk.FreqDist(bigrams)
    processed=sentence
    for x in PROCESS:
        processed=processed.replace(x,"")
       
    #adding each review into doc for count vectorization
    doc.append(processed)

    #word_count
    words=sentence.split()
    length=len(words)
    feature_list.append(length)
    #Distinct_WordCount    
    for w in words:
        if w in distinct: 
            distinct[w]=distinct[w]+1
        else:
            distinct.update({w:1})
    feature_list.append(len(distinct))
    

    #Sequence of words extrction using POS tagger
    text = nltk.tokenize.word_tokenize(processed)
    temporary=nltk.pos_tag(text)
    i=0
    def sequence1(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'a' and w2 == 'lot' and w3 == 'of'): 
                i=i+1
    sequence1(temporary)
    pos_list.append(i)
    feature_list.append(i)
    i=0
    def sequence2(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'one' and w2 == 'of' and w3 == 'the'): 
                i=i+1
    sequence2(temporary)
    pos_list.append(i)
    feature_list.append(i)
    i=0
    def sequence3(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'be' and w2 == 'able' and w3 == 'to'): 
                i=i+1
    sequence3(temporary)
    pos_list.append(i)
    feature_list.append(i)
    i=0
    def sequence4(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'a' and w2 == 'couple' and w3 == 'of'): 
                i=i+1
    sequence4(temporary)
    pos_list.append(i)
    feature_list.append(i)
    i=0
    def sequence5(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'to' and w2 == 'be' and w3 == 'a'): 
                i=i+1
    sequence5(temporary)
    pos_list.append(i)
    feature_list.append(i)

    #importing gender of review
    if 'M' in str(row[1]):
        gender.append(0)
    else:
        gender.append(1)

    #Appending complete feature list to an array
    train_features.append(feature_list)

#Importing test data from SQL server
gender_key=[]
test_features=[]
docs=[]
cursor.execute('SELECT * from test;')

#Loop through each review
for row in cursor:
    features_list=[]
    pos_list=[]
    distinct={}
    row_pos=[]
    rx=re.compile(r'(.)\1{7,}')
    sentence=row[0].replace("&nbsp","")
    words=sentence.split()

    processed=sentence
    for x in PROCESS:
        processed=processed.replace(x,"")
    
    #adding each review into doc for count vectorization
    docs.append(sentence)
    
    #word_count
    length=len(words)
    features_list.append(length)

    #importing gender of review
    if 'M' in str(row[1]):
        gender_key.append(0)
    else:
        gender_key.append(1)
    
    #Distinct_WordCount
    for w in words:
        if w in distinct:
            distinct[w]=distinct[w]+1
        else:
            distinct.update({w:1})
    features_list.append(len(distinct))


    #Sequence of words extrction using POS tagger
    text = nltk.tokenize.word_tokenize(processed)
    temporary=nltk.pos_tag(text)
    i=0
    def sequence1(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'a' and w2 == 'lot' and w3 == 'of'): 
                i=i+1
    sequence1(temporary)
    pos_list.append(i)
    features_list.append(i)
    i=0
    def sequence2(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'one' and w2 == 'of' and w3 == 'the'): 
                i=i+1
    sequence2(temporary)
    pos_list.append(i)
    features_list.append(i)
    i=0
    def sequence3(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'be' and w2 == 'able' and w3 == 'to'): 
                i=i+1
    sequence3(temporary)
    pos_list.append(i)
    features_list.append(i)
    i=0
    def sequence4(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'a' and w2 == 'couple' and w3 == 'of'): 
                i=i+1
    sequence4(temporary)
    pos_list.append(i)
    features_list.append(i)
    i=0
    def sequence5(sentence):
       global i
       for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
            if (w1 == 'to' and w2 == 'be' and w3 == 'a'): 
                i=i+1
    sequence5(temporary)
    pos_list.append(i)
    features_list.append(i)  
    #Appending complete feature list to an array
    test_features.append(features_list)

#Extracting top 10 word sequence pattern
pos_sequence={}
for sentence in doc:
    text = nltk.tokenize.word_tokenize(sentence)
    trigrams=nltk.trigrams(text)
    fdist = nltk.FreqDist(trigrams)
    for k,v in fdist.items():
        if k in pos_sequence:
            pos_sequence[k]=pos_sequence[k]+v
        else:
            pos_sequence.update({k:v})
        #print k,v

print 'Top 10 word sequence pattern'            
d=Counter(pos_sequence)    
for k, v in d.most_common(10):
   print '%s: %i' % (k, v)

#Bag of Words
i=[1,2,3]
clf=svm.LinearSVC(dual=False)
print 'linear_SVC- Bag of Words'
for x in i:
    vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,
                             preprocessor = None,stop_words = 'english',
                             max_features = 400, ngram_range=(1, x)) 
    X = vectorizer.fit_transform(doc).toarray()
    Y=vectorizer.fit_transform(docs).toarray()
    clf=clf.fit(X,gender)
    result=clf.predict(Y)
    print 'ngram(1,',(x),')'
    print accuracy_score(gender_key,result)

print 'svc- bag of words'
clf=svm.SVC(C=100, gamma=0.0001)
for x in i:
    vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,
                             preprocessor = None,stop_words = 'english',
                             max_features = 400, ngram_range=(1, x)) 
    X = vectorizer.fit_transform(doc).toarray()
    Y=vectorizer.fit_transform(docs).toarray()
    clf=clf.fit(X,gender)
    result=clf.predict(Y)
    print 'ngram(1,',(x),')'
    print accuracy_score(gender_key,result)
clf = RandomForestClassifier()
print 'Random Forest - Bag of Words'
for x in i:
    vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,
                             preprocessor = None,stop_words = 'english',
                             max_features = 400, ngram_range=(1, x)) 
    X = vectorizer.fit_transform(doc).toarray()
    Y=vectorizer.fit_transform(docs).toarray()
    clf=clf.fit(X,gender)
    result=clf.predict(Y)
    print 'ngram(1,',(x),')'
    print accuracy_score(gender_key,result)

#Support Vector Classification
clf=svm.SVC(C=100, gamma=0.0001)
x,y = train_features,gender
clf =clf.fit(x,y)
result=clf.predict(test_features)
print 'SVC'
print accuracy_score(gender_key,result)

#Linear SVC
clf=svm.LinearSVC(dual=False)
x,y = train_features,gender
clf =clf.fit(x,y)
result=clf.predict(test_features)
print 'Linear Svc'
print accuracy_score(gender_key,result)

#naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(train_features, gender).predict(test_features)
print 'naive bayes'
print accuracy_score(gender_key,y_pred)

#SGD classifier
from sklearn import linear_model
clf = linear_model.SGDClassifier()
x,y = train_features,gender
clf =clf.fit(x,y)
result=clf.predict(test_features)
print 'SGD classifier'
print accuracy_score(gender_key,result)

#SGD classifier- using los=log
from sklearn import linear_model
clf = linear_model.SGDClassifier(loss='log')
x,y = train_features,gender
clf =clf.fit(x,y)
result=clf.predict(test_features)
print 'SGD classifier - using los=log'
print accuracy_score(gender_key,result)

#SGD classifier- using loss='modified_huber'
from sklearn import linear_model
clf = linear_model.SGDClassifier(loss='modified_huber')
x,y = train_features,gender
clf =clf.fit(x,y)
result=clf.predict(test_features)
print 'SGD classifier - using loss=modified_huber'
print accuracy_score(gender_key,result)

#SGD classifier- using penalty=l1
from sklearn import linear_model
clf = linear_model.SGDClassifier(penalty='l1')
x,y = train_features,gender
clf =clf.fit(x,y)
result=clf.predict(test_features)
print 'SGD classifier - using penalty=l1'
print accuracy_score(gender_key,result)

#Extra tree classifier
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
x,y = train_features,gender
clf =clf.fit(x,y)
result=clf.predict(test_features)
print 'Extra tree classifier'
print accuracy_score(gender_key,result)



#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(clf, train_features, gender)
print 'Decision tree classifier'
print scores.mean()

#Random Forest Classifier
clf = RandomForestClassifier()
x,y = train_features,gender
clf = clf.fit(x,y)
test=clf.predict(test_features)
print 'Random Forest Classifier'
print accuracy_score(gender_key,test)


#AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
x,y = train_features,gender
clf =clf.fit(x,y)
result=clf.predict(test_features)
print 'ADA Boost Accuracy'
print accuracy_score(gender_key,result)

#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
x,y = train_features,gender
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(x,y)
result=clf.score(test_features,gender_key)
print 'Gradient Boosting classifier'
print result
