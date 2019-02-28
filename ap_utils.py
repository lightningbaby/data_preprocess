import csv
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import collections
import numpy as np
from gensim.models import word2vec

nltk.download('punkt')
nltk.download('stopwords')
stemmer=nltk.stem.PorterStemmer()
english_stopwords = stopwords.words('english')
english_punctuations=['the','/','^','-','+','<','>','{','}','*','//',',','.',':',';','?','(',')','[',']','&','!','*','@','|','#','$','%','"',"'","''",'""','`','``','\'','\\','=','\'s','\\\\','``','--','\'\'']


def get_corpus(file_content):
    all_words = []
    i = 0
    label = set()
    label_list = []
    for line in file_content:
        txt_label = line[0]
        label.add(txt_label)
        label_list.append(txt_label)
        txt_content = line[1] + line[2]  # concate title and description
        all_words.append(txt_content)
        i += 1
    return all_words, label_list

def del_low_freq(words_list):
    copy_list=[]
    i=0
    # print('bef del-word_list length:',len(words_list))
    for item in words_list:
        # print('del low freq--{} freq:{}'.format(item[0],item[1]))
        if item[1]>7:
            copy_list.append(item[0])
        i+=1
    # print('after del-word_list length:', len(words_list))
    # print('copy_list:',copy_list)
    return copy_list

# take much time
def get_stemmed(word_list):

    disease_List = nltk.word_tokenize(str(word_list).lower())
    filtered_txt = [w for w in disease_List if (w not in english_stopwords)]
    filtered_txt = [w for w in filtered_txt if (w not in english_punctuations)]  # del punctuations
    stemmed_txt = [w for w in filtered_txt if (w.isalpha())]
    stemmed_txt = [stemmer.stem(w) for w in stemmed_txt]
    stemmed_txt = [w for w in stemmed_txt if (w not in english_stopwords)]
    stemmed_txt = [w for w in stemmed_txt if (w not in english_punctuations)]
    # print('stemmed', stemmed_txt)

    return stemmed_txt

def get_each_w(word_list,name):
    each_word_list=[]
    i = 0
    for line in word_list:
        print('正在获取{} 文章 {}/{} 的all_words_freq...'.format(name,i,len(word_list)))
        stemmed_txt = get_stemmed(line)
        each_word_list.append(stemmed_txt)
        i += 1

    print('正在生成{}_each_word_list.txt...'.format(name))
    name1=name+'_each_word_list.txt'
    np.savetxt(name1, each_word_list, fmt='%s', delimiter=' ', newline='\n')


    return each_word_list

def get_word_id(word,max_wordlist):
    if word in max_wordlist:
        return max_wordlist.index(word)
    else:
        return False

def get_features_matrix(txt_num, maxlength,out_dims,each_wordlist,model,name):
    features=np.zeros((txt_num,maxlength,out_dims)).astype(float)
    ap_features=np.zeros((txt_num,1,out_dims)).astype(float)
    word_arr=np.zeros((maxlength,out_dims)).astype(float)
    temp=np.zeros((txt_num,maxlength,out_dims))
    txt_cnt=0
    # print('each word list:',each_wordlist)
    for txt in each_wordlist:
        print('正在生成{} 的特征矩阵 文章{}/{}'.format(name,txt_cnt,txt_num))
        word_cnt=0
        wrong_word_cnt=0
        word_id_cnt=0
        for w in txt:
            try:
                # print('txt cnt={} , word cnt={}/{}, {} '.format(txt_cnt,word_cnt,maxlength, w))
                if(word_id_cnt<=maxlength):
                    features[txt_cnt, word_id_cnt, :] = model[w]
                    word_arr[word_id_cnt,:]=model[w]
                    # print('model[{}].all={}'.format(w,model[w]))
                    # print('features[{},{},]={}'.format(txt_cnt,word_id_cnt,features[txt_cnt,word_id_cnt,:]))
                else:
                    print('word id cnt:{} is more than maxlength.'.format(word_id_cnt))

            except KeyError:
                # print('txt cnt={} ,wrong id:{}, {} not in voc'.format(txt_cnt,wrong_word_cnt,w))
                wrong_word_cnt+=1
            word_cnt+=1

        average_arr=word_arr.sum(axis=0) / word_cnt
        # print('average arr:',average_arr.shape)
        # print(average_arr)
        ap_features[txt_cnt,0,:]=average_arr
        # print('ap features:',ap_features.shape)
        # print(ap_features)

        txt_cnt+=1
    if (features==temp).all():
        print('features all zeros')
    else:
        print('features right')

    return ap_features

def get_corpus_file(all_words):
    stemmed=get_stemmed(all_words)
    all_words_freq = collections.Counter(stemmed).most_common()
    all_words_nofreq = del_low_freq(all_words_freq)

    np.savetxt('ap_all_words_nofreq.txt', all_words_nofreq, fmt='%s', delimiter=' ', newline='\n')
    corpus_name='ap_all_fil.txt'
    np.savetxt(corpus_name,stemmed,fmt='%s',delimiter=' ',newline=' ')

def get_graph(model):
    words = open('ap_all_words_nofreq.txt', 'r')
    # 这里读出来的数据会有格式问题，所有重新append一下
    word_list = []
    for word in words:
        word = word.strip('\n')
        word_list.append(word)
        # print(word)

    length = len(word_list)
    # np.savetxt('bow_gcn_all_words_nofreq_len.txt', length)
    print('ap all no freq len:', length)
    # print('word list:',word_list)
    graph = np.zeros([length, length], dtype='float')
    check_arr = np.zeros([length, length], dtype='float')
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
                    not_in_wordlist += 1
                    print('{} {}/{} not in word list'.format(top[w][0], not_in_wordlist, length))
                    # id=get_word_id(top[w][0], word_list)
                    # print('id :',id)
        except KeyError:
            print('{} {}/{} not in model'.format(word_list[i], wrong_word_cnt, length))
            wrong_word_cnt = +1

    # 判读graph是否成功赋值
    if (graph == check_arr).all():
        print('graph worng')
    else:
        print('graph right')

    np.save('graph.npy', graph)
    print('graph:', graph.shape)
    # print(graph)
    return graph

def load_data():
    # read train csv
    file_content=csv.reader(open('./data/train.csv','r'))
    all_words,label_list=get_corpus(file_content)


    # split train csv into train part and val part
    train_words,val_words,train_label,val_label=train_test_split(all_words,label_list,test_size=0.1)


    # each word in train and val
    train_each_wordlist=get_each_w(train_words,'train')
    val_each_wordlist=get_each_w(val_words,'val')

    # find max_word_list in train part
    maxlength=0
    # max_wordlist=[]
    for i in train_each_wordlist:
        # print('{} in train each word list.'.format(i))
        if len(i)>maxlength:
            maxlength=len(i)
            # max_wordlist=i
    print('max length:',maxlength)
    # print('max word list:',max_wordlist)
    # np.savetxt('max_word_list.txt',max_wordlist,fmt='%s',delimiter=' ',newline=' ')

    # get corpus
    print('正在生成语料文件...')
    get_corpus_file(train_words)

    # train word2vec
    out_dims=50
    print('正在训练语料...')
    sentences = word2vec.Text8Corpus(u'ap_all_fil.txt')
    model = word2vec.Word2Vec(sentences, size=out_dims,min_count=8)
    model.save(u'txt.model')

    # construct graph
    graph=get_graph(model)

    # construct features
    # define features matrix=(txt_num, len(max_word_list),out_dims)
    print('正在生成特征矩阵...')
    train_features=get_features_matrix(len(train_label),maxlength,out_dims,train_each_wordlist,model,'train')
    val_features=get_features_matrix(len(val_label),maxlength,out_dims,val_each_wordlist,model,'val')


    # read test
    # split and stemming
    # get features
    file_content=csv.reader(open('./data/test.csv','r'))
    test_all_words,test_label=get_corpus(file_content)
    test_each_wordlist=get_each_w(test_all_words,'test')
    test_features=get_features_matrix(len(test_label),maxlength,out_dims,test_each_wordlist,model,'test')

    # match interface
    train_label=np.array(train_label)
    val_label=np.array(val_label)
    test_label=np.array(test_label)

    y_train=np.zeros((len(train_each_wordlist)))
    y_val=np.zeros((len(val_each_wordlist)))
    y_test=np.zeros((len(test_each_wordlist)))

    np.save('./data/ap_graph.npy',graph)
    np.save('./data/ap_train_features.npy',train_features)
    np.save('./data/ap_train_label.npy',train_label)
    np.save('./data/ap_y_train.npy',y_train)

    np.save('./data/ap_val_features.npy', val_features)
    np.save('./data/ap_val_label.npy', val_label)
    np.save('./data/ap_y_val.npy', y_val)

    np.save('./data/ap_test_features.npy', test_features)
    np.save('./data/ap_test_label.npy', test_label)
    np.save('./data/ap_y_test.npy', y_test)

    return graph,train_features,train_label,y_train,val_features,val_label,y_val,test_features,test_label,y_test


def load_data_from_file():
    print('loading data from file...')
    graph=np.load('./data/ap_graph.npy')
    train_features=np.load('./data/ap_train_features.npy')
    train_label=np.load('./data/ap_train_label.npy')
    y_train=np.load('./data/ap_y_train.npy')

    val_features=np.load('./data/ap_val_features.npy')
    val_label=np.load('./data/ap_val_label.npy')
    y_val=np.load('./data/ap_y_val.npy')

    test_features=np.load('./data/ap_test_features.npy')
    test_label=np.load('./data/ap_test_label.npy')
    y_test=np.load('./data/ap_y_test.npy')

    return graph,train_features,train_label,y_train,val_features,val_label,y_val,test_features,test_label,y_test

# graph,train_features,train_label,y_train,val_features,val_label,y_val,test_features,test_label,y_test=load_data_from_file()
# #
# print('graph:',graph.shape)
# # # print(graph)
# # # #
# print('train features:',train_features.shape)
# # print('train label:',train_label.shape)
# # print('y train:',y_train.shape)
# # #
# print('train features:',train_features)
# # print('train label:',train_label)
# # print('y train:',y_train)
# # # #
# # # #
# print('val features:',val_features.shape)
# # print('val label:',val_label.shape)
# # print('y val:',y_val.shape)
# # #
# print('val features:',val_features)
# # print('val label:',val_label)
# # print('y val:',y_val)
# # #
# print('test features:',test_features.shape)
# # print('test label:',test_label.shape)
# # print('y test:',y_test.shape)
# # #
# print('test features:',test_features)
# # print('test label:',test_label)
# # print('y test:',y_test)




