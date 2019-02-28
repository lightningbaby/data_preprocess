import csv
import sklearn.feature_extraction.text as tfidf
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import collections
from gensim.models import word2vec



def get_corpus(file_content):
    all_words = []
    i = 0
    label = set()
    label_list=[]
    for line in file_content:
        txt_label = line[0]
        label.add(txt_label)
        label_list.append(txt_label)
        txt_content = line[1] + line[2]  # concate title and description
        all_words.append(txt_content)
        i+=1
    label_list=np.array(label_list).astype(int)
    return all_words,label_list

def get_tfidf_arr(all_words):
    vectorizer = tfidf.TfidfVectorizer(lowercase=True, stop_words='english', min_df=7)# min_df 设置和del_low_freq中的一样
    vectorizer.fit_transform(all_words)
    words = vectorizer.get_feature_names()
    print('get tfidf arr words:',len(words))
    arr = vectorizer.fit_transform(all_words).toarray()
    arr = np.array(arr)
    all_words_freq=vectorizer.vocabulary_
    # print(all_words)
    # np.savetxt('bow_gcn_tfidf_len.txt',len(words))
    return arr,all_words_freq



def get_word_id(word,word_list):
    # 这里有个陷阱
    # 如果返回false，则 当id是0的时候，也会被判断为false，即不在word_list中
    try:
        word_id=word_list.index(word)
        return word_id
    except ValueError:
        print('{} not in word list'.format(word))
        return -1

def word_vec():
    sentences = word2vec.Text8Corpus(u'bow_all_fil.txt')  # 加载语料
    model = word2vec.Word2Vec(sentences, min_count=0)  # 训练skip-gram模型，默认window=5
    model.save(u'txt.model')
    print('完成训练')
    print('model:',model.wv.vocab)

    words = open('bow_word_table.txt', 'r')
    # 这里读出来的数据会有格式问题，所有重新append一下
    word_list = []
    for word in words:
        word = word.strip('\n')
        word_list.append(word)
        # print(word)

    length = len(word_list)
    print('bow all no freq len:', length)
    # print('word list:',word_list)
    graph = np.zeros([length, length], dtype='float')
    check_arr=np.zeros([length, length], dtype='float')
    for i in range(length):
        wrong_word_cnt = 0
        not_in_wordlist = 0
        try:
            top = model.most_similar(word_list[i], topn=16)
            # print('top:',top)
            for w in range(len(top)):
                word_id = get_word_id(top[w][0], word_list)
                if word_id != -1:
                    graph[i][word_id] = top[w][1]
                else:
                    # word2vec 训练的语料bow_all_freq.txt和bow_all_nofreq.txt都是通过nltk得到，
                    # 后者在前者的基础上删除了低频词，低频阈值和Word2vec设置的一样
                    # 但即使这样，Word2vec训练得到的模型，生成的相似度大的词，可能不存在bow_all_nofreq.txt中
                    # 可能是Word2vec的过滤方式不一样导致的
                    # 将Word2vec的阈值设置的大于del_low_freq 1个
                    # 这样word2vec 训练的模型的词和 bow_all_nofreq中的词一致
                    not_in_wordlist +=1
                    print('{} {}/{} not in word list'.format(top[w][0],not_in_wordlist,length))
                    # id=get_word_id(top[w][0], word_list)
                    # print('id :',id)
        except KeyError:
            wrong_word_cnt = +1
            print('{} {}/{} not in model'.format(word_list[i], wrong_word_cnt, length))


    # 判读graph是否成功赋值
    if (graph==check_arr).all():
        print('graph worng')
    else:
        print('graph right')

    np.save('graph.npy', graph)
    print('graph:', graph.shape)
    # print(graph)
    return graph



# read file
# get corpus
def load_data(dataset,type,name):
    filename = '../data/' + dataset + '_train.csv'
    train_filecontent = csv.reader(open(filename, 'r'))
    all_words, label_list = get_corpus(train_filecontent)

# get tfidf arr
# return voc
    tfidf_arr,all_words_freq=get_tfidf_arr(all_words) #min 7
    word_tabel=[w for w,v in all_words_freq.items()]
    print('词表 len:{},{}'.format(len(word_tabel),word_tabel))
    np.savetxt('bow_word_table.txt',word_tabel,fmt='%s',delimiter=' ',newline='\n')
    np.savetxt('bow_all_fil.txt', word_tabel, fmt='%s', delimiter=' ', newline=' ')

# train word2vec with all fil
    graph=word_vec()

# process corpus with nltk
# save all fil txt


# train word2vec with above all fil txt


# compare tfidf's voc with above all fil
    return graph


graph=load_data('easy','train','bow')