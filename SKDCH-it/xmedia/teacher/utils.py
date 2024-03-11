# -- coding: utf-8--
import linecache
import numpy as np

def list_split(items):
    return [items[i: i + 1] for i in range(0, len(items), 1)]

def is_same_cate(strA, strB, label_dim):
    labelA = list_split(strA)
    labelB = list_split(strB)
    
    for i in range(label_dim):
        if labelA[i] == [1] and labelA[i] == labelB[i]:
            return True
    return False


def push_query(query, url, dict):
    if query[0] in dict:
        dict[query[0]].append(url[0])
    else:
        dict[query[0]] = [url[0]]
    return dict


def make_train_dict(query_list, url_list, semi_list, query_label, url_label, label_dim):
    query_url = {}
    query_pos = {}
    query_neg = {}
    query_pair = {}
    query_num = len(query_list)
    url_num = len(url_list)
    semi_num = len(semi_list)
    
    for i in range(query_num):
        query = query_list[i]
        for j in range(url_num):
            url = url_list[j]
            if i == j:
                push_query(query, url, query_url)
                push_query(query, url, query_pos)
                push_query(query, url, query_pair)
            elif is_same_cate(query_label[i], url_label[j], label_dim):
                push_query(query, url, query_url)
                push_query(query, url, query_pos)
            else:
                push_query(query, url, query_url)
                push_query(query, url, query_neg)
        for j in range(semi_num):
            url = semi_list[j]
            push_query(query, url, query_url)
    
    return query_url, query_pos, query_neg, query_pair


def make_test_dict(query_list, url_list, query_label, url_label, label_dim):
    query_url = {}
    query_pos = {}
    query_num = len(query_list)
    url_num = len(url_list)
    
    for i in range(query_num):
        query = query_list[i]
        for j in range(url_num):
            url = url_list[j]
            if is_same_cate(query_label[i], url_label[j], label_dim):
                push_query(query, url, query_url)
                push_query(query, url, query_pos)
            else:
                push_query(query, url, query_url)
    return query_url, query_pos


def load_all_query_url(list_dir, label_dim):
    train_img = np.load(list_dir + 'train_img_path_list.npy')
    semi_img = np.load(list_dir + 'semi_img_path_list.npy')
    test_img = np.load(list_dir + 'test_img_path_list.npy')
    database_img = np.load(list_dir + 'database_img_path_list.npy')
    
    train_txt = np.load(list_dir + 'train_txt_path_list.npy')
    semi_txt = np.load(list_dir + 'semi_txt_path_list.npy')
    test_txt = np.load(list_dir + 'test_txt_path_list.npy')
    database_txt = np.load(list_dir + 'database_txt_path_list.npy')
    
    train_img_label = np.load(list_dir + 'labels_train_img.npy')
    train_txt_label = np.load(list_dir + 'labels_train_txt.npy')
    test_img_label = np.load(list_dir + 'labels_test_img.npy')
    test_txt_label = np.load(list_dir + 'labels_test_txt.npy')
    database_img_label = np.load(list_dir + 'labels_database_img.npy')
    database_txt_label = np.load(list_dir + 'labels_database_txt.npy')
    
    train_t2i, train_t2i_pos, train_t2i_neg, train_t2i_pair = make_train_dict(train_txt, train_img, semi_img,
                                                                              train_txt_label, train_img_label,
                                                                              label_dim)
    
    train_i2t, train_i2t_pos, train_i2t_neg, train_i2t_pair = make_train_dict(train_img, train_txt, semi_txt,
                                                                              train_img_label, train_txt_label,
                                                                              label_dim)
    
    test_t2i, test_t2i_pos = make_test_dict(test_txt, database_img, test_txt_label, database_img_label, label_dim)
    test_i2t, test_i2t_pos = make_test_dict(test_img, database_txt, test_img_label, database_txt_label, label_dim)
    
    print('Training num %d' % len(train_t2i))
    print('Testing num %d' % len(test_t2i))
    
    return train_t2i, train_t2i_pos, train_t2i_neg, train_t2i_pair, \
           train_i2t, train_i2t_pos, train_i2t_neg, train_i2t_pair, \
           test_t2i, test_t2i_pos, \
           test_i2t, test_i2t_pos


def load_all_feature_for_train_2(list_dir):
    train_img_list = np.load(list_dir + 'train_img_path_list_list.npy').tolist()
    semi_img_list = np.load(list_dir + 'semi_img_path_list_list.npy').tolist()
    test_img_list = np.load(list_dir + 'test_img_path_list_list.npy').tolist()
    
    train_txt_list = np.load(list_dir + 'train_txt_path_list_list.npy').tolist()
    semi_txt_list = np.load(list_dir + 'semi_txt_path_list_list.npy').tolist()
    test_txt_list = np.load(list_dir + 'test_txt_path_list_list.npy').tolist()
    
    train_img = np.load(list_dir + 'img_train.npy').tolist()
    semi_img = np.load(list_dir + 'img_semi.npy').tolist()
    test_img = np.load(list_dir + 'img_test.npy').tolist()
    
    train_txt = np.load(list_dir + 'texts_train.npy').tolist()
    semi_txt = np.load(list_dir + 'texts_semi.npy').tolist()
    test_txt = np.load(list_dir + 'texts_test.npy').tolist()
    
    feature_train_img_dict = dict(zip(train_img_list, train_img))
    feature_semi_img_dict = dict(zip(semi_img_list, semi_img))
    feature_test_img_dict = dict(zip(test_img_list, test_img))
    
    feature_train_txt_dict = dict(zip(train_txt_list, train_txt))
    feature_semi_txt_dict = dict(zip(semi_txt_list, semi_txt))
    feature_test_txt_dict = dict(zip(test_txt_list, test_txt))
    
    feature_train_img_dict.update(feature_semi_img_dict)
    feature_train_img_dict.update(feature_test_img_dict)
    feature_train_img_dict.update(feature_train_txt_dict)
    feature_train_img_dict.update(feature_semi_txt_dict)
    feature_train_img_dict.update(feature_test_txt_dict)
    
    feature_dict = feature_train_img_dict
    return feature_dict


def load_all_feature_for_train_test_2(list_dir):
    test_img_list = np.load(list_dir + 'test_img_path_list_list.npy').tolist()
    database_img_list = np.load(list_dir + 'database_img_path_list_list.npy').tolist()
    
    test_txt_list = np.load(list_dir + 'test_txt_path_list_list.npy').tolist()
    database_txt_list = np.load(list_dir + 'database_txt_path_list_list.npy').tolist()
    
    database_img = np.load(list_dir + 'img_database.npy').tolist()
    test_img = np.load(list_dir + 'img_test.npy').tolist()
    
    database_txt = np.load(list_dir + 'texts_database.npy').tolist()
    test_txt = np.load(list_dir + 'texts_test.npy').tolist()
    
    feature_test_img_dict = dict(zip(test_img_list, test_img))
    feature_test_txt_dict = dict(zip(test_txt_list, test_txt))
    
    feature_database_img_dict = dict(zip(database_img_list, database_img))
    feature_database_txt_dict = dict(zip(database_txt_list, database_txt))
    
    feature_test_img_dict.update(feature_test_txt_dict)
    feature_test_img_dict.update(feature_database_img_dict)
    feature_test_img_dict.update(feature_database_txt_dict)
    
    feature_dict = feature_test_img_dict
    return feature_dict


def load_all_feature_for_train(list_dir, feature_dir):
    feature_dict = {}
    for dataset in ['train', 'test', 'semi']:
        for modal in ['txt', 'img']:
            list = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\r\n')
            feature = open(feature_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
            
            for i in range(len(list) - 1):
                item = list[i]
                feature_string = feature[i].split()
                feature_float_list = []
                for j in range(len(feature_string)):
                    feature_float_list.append(float(feature_string[j]))
                feature_dict[item] = feature_float_list
    return feature_dict


def load_all_feature_for_train_test(list_dir, feature_dir):
    feature_dict = {}
    for dataset in ['database', 'test']:
        for modal in ['txt', 'img']:
            list = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
            feature = open(feature_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
            
            for i in range(len(list) - 1):
                item = list[i]
                feature_string = feature[i].split()
                feature_float_list = []
                for j in range(len(feature_string)):
                    feature_float_list.append(float(feature_string[j]))
                feature_dict[item] = feature_float_list
    return feature_dict


def load_all_feature_for_test(list_dir, feature_dir):
    feature_dict = {}
    for dataset in ['database', 'test']:
        for modal in ['txt', 'img']:
            list = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
            feature = open(feature_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
            
            for i in range(len(list) - 1):
                item = list[i]
                feature_string = feature[i].split()
                feature_float_list = []
                for j in range(len(feature_string)):
                    feature_float_list.append(float(feature_string[j]))
                feature_dict[item] = feature_float_list
    return feature_dict


def load_all_label(list_dir):
    label_dict = {}
    for dataset in ['train', 'database', 'test', 'semi']:
        for modal in ['txt', 'img']:
            list = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
            label = open(list_dir + dataset + '_' + modal + '_label.txt', 'r').read().split('\n')
            for i in range(len(list) - 1):
                item = list[i]
                label_string = label[i].split()
                label_float_list = []
                for j in range(len(label_string)):
                    label_float_list.append(float(label_string[j]))
                label_dict[item] = label_float_list
    return label_dict


def load_all_label_2(list_dir):
    train_img_label = np.load(list_dir + 'labels_train_img.npy').tolist()  # [标签]
    train_txt_label = np.load(list_dir + 'labels_train_txt.npy').tolist()
    semi_img_label = np.load(list_dir + 'labels_semi_img.npy').tolist()
    semi_txt_label = np.load(list_dir + 'labels_semi_txt.npy').tolist()
    validation_img_label = np.load(list_dir + 'labels_database_img.npy').tolist()
    validation_txt_label = np.load(list_dir + 'labels_database_txt.npy').tolist()
    test_img_label = np.load(list_dir + 'labels_test_img.npy').tolist()
    test_txt_label = np.load(list_dir + 'labels_test_txt.npy').tolist()
    
    train_label_list_img = np.load(list_dir + 'train_img_path_list_list.npy').tolist()  # [名字list]
    semi_label_list_img = np.load(list_dir + 'semi_img_path_list_list.npy').tolist()
    test_label_list_img = np.load(list_dir + 'test_img_path_list_list.npy').tolist()
    validation_label_list_img = np.load(list_dir + 'database_img_path_list_list.npy').tolist()
    
    train_label_list_txt = np.load(list_dir + 'train_txt_path_list_list.npy').tolist()
    semi_label_list_txt = np.load(list_dir + 'semi_txt_path_list_list.npy').tolist()
    test_label_list_txt = np.load(list_dir + 'test_txt_path_list_list.npy').tolist()
    validation_label_list_txt = np.load(list_dir + 'database_txt_path_list_list.npy').tolist()
    
    train_img_label_dict = dict(zip(train_label_list_img, train_img_label))
    semi_img_label_dict = dict(zip(semi_label_list_img, semi_img_label))
    validation_img_label_dict = dict(zip(validation_label_list_img, validation_img_label))
    test_img_label_dict = dict(zip(test_label_list_img,
                                   test_img_label))
    
    train_img_label_dict.update(semi_img_label_dict)
    train_img_label_dict.update(validation_img_label_dict)
    train_img_label_dict.update(test_img_label_dict)
    img_label_dict = train_img_label_dict
    
    train_txt_label_dict = dict(zip(train_label_list_txt, train_txt_label))
    semi_txt_label_dict = dict(zip(semi_label_list_txt, semi_txt_label))
    validation_txt_label_dict = dict(zip(validation_label_list_txt, validation_txt_label))
    test_txt_label_dict = dict(zip(test_label_list_txt,test_txt_label))

    
    train_txt_label_dict.update(semi_txt_label_dict)
    train_txt_label_dict.update(validation_txt_label_dict)
    train_txt_label_dict.update(test_txt_label_dict)
    txt_label_dict = train_txt_label_dict

    img_label_dict.update(txt_label_dict)
    label_dict = img_label_dict
    return label_dict


def get_query_pos(file, semi_flag):
    query_pos = {}
    with open(file) as fin:
        for line in fin:
            cols = line.split()
            rank = float(cols[0])
            query = cols[1]
            url = cols[2]
            if rank > semi_flag:
                if query in query_pos:
                    query_pos[query].append(url)
                else:
                    query_pos[query] = [url]
    return query_pos


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_batch_data(file, index, size):
    pos = []
    neg = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        line = line.strip().split()
        pos.append([float(x) for x in line[0].split(',')])
        neg.append([float(x) for x in line[1].split(',')])
    return pos, neg
