import numpy as np

def get_file_name(dataset,type,name):
    if type=='train':
        # label_class_file = '../data/gcn_' + dataset + '_label_num.npy'
        graph_file = './data/' +name+'_'+ dataset + '_graph.npy'

        train_features_file =  './data/' +name+'_'+ dataset + '_train_features.npy'
        train_label_file =  './data/' +name+'_'+ dataset + '_train_label.npy'
        y_train_file = './data/' +name+'_'+ dataset + '_y_train.npy'

        val_features_file =  './data/' +name+'_'+ dataset + '_val_features.npy'
        val_label_file =  './data/' +name+'_'+ dataset + '_val_label.npy'
        y_val_file =  './data/' +name+'_'+ dataset + '_y_val.npy'
        return graph_file, train_features_file, train_label_file, \
               y_train_file, val_features_file, val_label_file, y_val_file,
    else:
        test_features_file = './data/' +name+'_'+ dataset + '_test_features.npy'
        test_label_file =  './data/' +name+'_'+ dataset + '_test_label.npy'
        y_test_file =  './data/' +name+'_'+ dataset + '_y_test.npy'

        return test_features_file, test_label_file,y_test_file


def load_data_from_file(dataset,type,name):
    # dataset是数据名
    # type 区分 train，test
    # name 是实验名字
    print('loading data from file...')
    if type=='train':
        graph_file, train_features_file, train_label_file, \
        y_train_file, val_features_file, val_label_file, y_val_file=get_file_name(dataset,type,name)

        # label_class=np.load(label_class_file)
        graph = np.load(graph_file)
        train_features = np.load(train_features_file)
        train_label = np.load(train_label_file)
        y_train = np.load(y_train_file)

        val_features = np.load(val_features_file)
        val_label = np.load(val_label_file)
        y_val = np.load(y_val_file)

        return graph, train_features, train_label, y_train, val_features, \
               val_label, y_val
    else:
        test_features_file, \
        test_label_file, y_test_file = get_file_name(dataset,type,name)

        test_features = np.load(test_features_file)
        test_label = np.load(test_label_file)
        y_test = np.load(y_test_file)

    return test_features, test_label, y_test