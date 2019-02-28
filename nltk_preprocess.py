import numpy as np
import nltk
from nltk.corpus import stopwords
import collections
from gensim.models import word2vec

nltk.download('punkt')
nltk.download('stopwords')
stemmer=nltk.stem.PorterStemmer()
english_stopwords = stopwords.words('english')
english_punctuations=['ap','reuters','reuter','sadi','st.','the','/','^','-','+','<','>','{','}','*','//',',','.',':',';','?','(',')','[',']','&','!','*','@','|','#','$','%','"',"'","''",'""','`','``','\'','\\','=','\'s','\\\\','``','--','\'\'']

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

def get_all_fil_freq(all_words):
    disease_List = nltk.word_tokenize(str(all_words).lower())
    filtered_txt = [w for w in disease_List if (w not in english_stopwords)]
    filtered_txt = [w for w in filtered_txt if (w not in english_punctuations)]  # del punctuations
    stemmed_txt = [stemmer.stem(w) for w in filtered_txt]
    stemmed_txt = [w for w in stemmed_txt if (w.isalpha())]
    # isalpha 可以过滤掉大多数不规范的字符，判断是不是全是字符
    # isdigit 判断是不是数字
    stemmed_txt = [w for w in stemmed_txt if (w not in english_stopwords)]
    stemmed_txt = [w for w in stemmed_txt if (w not in english_punctuations)]
    # print('stemmed', stemmed_txt)
    all_words_freq = collections.Counter(stemmed_txt).most_common()
    print('all freq len:',len(all_words_freq))
    all_words_nofreq=del_low_freq(all_words_freq)
    print('all freq len after del low freq:', len(all_words_nofreq))


    np.savetxt('bow_all_fil.txt',stemmed_txt,fmt='%s',delimiter=' ',newline=' ')
    np.savetxt('all_words_nofreq.txt',all_words_nofreq,fmt='%s',delimiter=' ',newline='\n')
