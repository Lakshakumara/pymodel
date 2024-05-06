import pickle

with open('./lr_model1.pickle', 'rb') as f:
    tfidf = pickle.load(f)

with open('./lr_model1.pickle', 'rb') as f:
    clf = pickle.load(f)

x = input("Please enter your phrase: ")
y = clf.predict_proba(tfidf.transform([x]))
print(y[0])