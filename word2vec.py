from gensim.models import word2vec
import numpy as np

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
    model = word2vec.Word2Vec(sentences, min_count=8)  # 训练skip-gram模型，默认window=5
    model.save(u'txt.model')
    print('完成训练')

    words = open('all_words_nofreq.txt', 'r')
    # 这里读出来的数据会有格式问题，所有重新append一下
    word_list = []
    for word in words:
        word = word.strip('\n')
        word_list.append(word)
        # print(word)

    length = len(word_list)
    np.savetxt('bow_gcn_all_words_nofreq_len.txt',length)
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
            print('{} {}/{} not in model'.format(word_list[i], wrong_word_cnt, length))
            wrong_word_cnt = +1

    # 判读graph是否成功赋值
    if (graph==check_arr).all():
        print('graph worng')
    else:
        print('graph right')

    np.save('graph.npy', graph)
    print('graph:', graph.shape)
    # print(graph)
    return graph