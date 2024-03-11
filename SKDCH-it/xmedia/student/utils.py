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

def make_train_dict(query_list, url_list, query_label, url_label, label_dim):
    query_url = {}
    query_pos = {}
    query_neg = {}
    query_pair = {}
    query_num = len(query_list)
    url_num = len(url_list)

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
    train_img = np.load(list_dir + 'retrieval_img_path_list.npy')[0:5000]
    test_img = np.load(list_dir + 'test_img_path_list.npy')
    database_img = np.load(list_dir + 'database_img_path_list.npy')
    
    train_txt = np.load(list_dir + 'retrieval_txt_path_list.npy')[0:5000]
    test_txt = np.load(list_dir + 'test_txt_path_list.npy')
    database_txt = np.load(list_dir + 'database_txt_path_list.npy')
    
    train_img_label = np.load(list_dir + 'I2T_train_L.npy')
    train_txt_label = np.load(list_dir + 'I2T_train_L.npy')
    
    test_img_label = np.load(list_dir + 'labels_test_img.npy')
    test_txt_label = np.load(list_dir + 'labels_test_txt.npy')
    database_img_label = np.load(list_dir + 'labels_database_img.npy')
    database_txt_label = np.load(list_dir + 'labels_database_txt.npy')
    
    train_t2i, train_t2i_pos, train_t2i_neg, train_t2i_pair = make_train_dict(train_txt, train_img,
                                                                              train_txt_label, train_img_label,
                                                                              label_dim)
    
    train_i2t, train_i2t_pos, train_i2t_neg, train_i2t_pair = make_train_dict(train_img, train_txt,
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
    train_img_list = np.load(list_dir + 'retrieval_img_path_list_list.npy')[0:5000].tolist()

    test_img_list = np.load(list_dir + 'test_img_path_list_list.npy').tolist()
    
    train_txt_list = np.load(list_dir + 'retrieval_txt_path_list_list.npy')[0:5000].tolist()

    test_txt_list = np.load(list_dir + 'test_txt_path_list_list.npy').tolist()
    
    train_img = np.load(list_dir + 'img_retrieval.npy')[0:5000].tolist()

    test_img = np.load(list_dir + 'img_test.npy').tolist()
    
    train_txt = np.load(list_dir + 'texts_retrieval.npy')[0:5000].tolist()
 
    test_txt = np.load(list_dir + 'texts_test.npy').tolist()
    
    feature_train_img_dict = dict(zip(train_img_list, train_img))

    feature_test_img_dict = dict(zip(test_img_list, test_img))
    
    feature_train_txt_dict = dict(zip(train_txt_list, train_txt))

    feature_test_txt_dict = dict(zip(test_txt_list, test_txt))
    

    feature_train_img_dict.update(feature_test_img_dict)
    feature_train_img_dict.update(feature_train_txt_dict)

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



def load_all_label_2(list_dir):
    train_img_label = np.load(list_dir + 'I2T_train_L.npy').tolist()
    train_txt_label = np.load(list_dir + 'I2T_train_L.npy').tolist()

    validation_img_label = np.load(list_dir + 'labels_database_img.npy').tolist()
    validation_txt_label = np.load(list_dir + 'labels_database_txt.npy').tolist()
    test_img_label = np.load(list_dir + 'labels_test_img.npy').tolist()
    test_txt_label = np.load(list_dir + 'labels_test_txt.npy').tolist()
    
    train_label_list_img = np.load(list_dir + 'retrieval_img_path_list_list.npy')[0:5000].tolist()
 
    test_label_list_img = np.load(list_dir + 'test_img_path_list_list.npy').tolist()
    validation_label_list_img = np.load(list_dir + 'database_img_path_list_list.npy').tolist()
    
    train_label_list_txt = np.load(list_dir + 'retrieval_txt_path_list_list.npy')[0:5000].tolist()

    test_label_list_txt = np.load(list_dir + 'test_txt_path_list_list.npy').tolist()
    validation_label_list_txt = np.load(list_dir + 'database_txt_path_list_list.npy').tolist()
    
    train_img_label_dict = dict(zip(train_label_list_img, train_img_label))

    validation_img_label_dict = dict(zip(validation_label_list_img, validation_img_label))
    test_img_label_dict = dict(zip(test_label_list_img,
                                   test_img_label))
    train_img_label_dict.update(validation_img_label_dict)
    train_img_label_dict.update(test_img_label_dict)
    img_label_dict = train_img_label_dict
    train_txt_label_dict = dict(zip(train_label_list_txt, train_txt_label))

    validation_txt_label_dict = dict(zip(validation_label_list_txt, validation_txt_label))
    test_txt_label_dict = dict(zip(test_label_list_txt,
                                   test_txt_label))
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
#
