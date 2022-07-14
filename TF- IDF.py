Document1 = "It is going to rain today."
Document2 = "TOday I am not going outside"
Document3 = "I am going to watch the season premiere"

Doc = [Document1, Document2, Document3]
print(Doc)

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd

vectorizer = TfidfVectorizer()

analyze = vectorizer.build_analyzer()

print('Document 1', analyze(Document1))
print('Document 2', analyze(Document2))
print('Document 3', analyze(Document3))

x= vectorizer.fit_transform(Doc)

print('Document transform',x.toarray())

print(vectorizer.get_feature_names())

