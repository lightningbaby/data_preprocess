import sklearn.feature_extraction.text as tfidf
import numpy as np

def get_tfidf_arr(all_words):
    vectorizer = tfidf.TfidfVectorizer(lowercase=True, stop_words='english', min_df=7)# min_df 设置和del_low_freq中的一样
    vectorizer.fit_transform(all_words)
    words = vectorizer.get_feature_names()
    print('get tfidf arr words:',len(words))
    arr = vectorizer.fit_transform(all_words).toarray()
    arr = np.array(arr)
    all_words_freq=vectorizer.vocabulary_
    # print(all_words)
    np.savetxt('bow_gcn_tfidf_len.txt',len(words))
    return arr