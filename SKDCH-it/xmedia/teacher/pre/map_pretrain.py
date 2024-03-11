# coding=UTF-8
import numpy as np

def calcu_map(query_pos_test, query_index_url_test, hash_dict, label_dict):
    map = 0
    for query in query_pos_test.keys():
        pred_list = query_index_url_test[query]
        query_hash = hash_dict[query]
        query_L = label_dict[query]

        code_length = query_hash.shape[0]
        candidates_hash = []
        retrieval_L = []
        for candidate in pred_list:
            candidates_hash.append(hash_dict[candidate])
            retrieval_L.append(label_dict[candidate])
        candidates_hash = np.asarray(candidates_hash)
        retrieval_L = np.asarray(retrieval_L)
        pred_list_score = code_length - np.sum(np.bitwise_xor(query_hash, candidates_hash), axis=1)
        idx = np.argsort(-pred_list_score)
        gnd = (np.dot(query_L, retrieval_L.transpose()) > 0).astype(np.float32)
        gnd = gnd[idx]
        tsum = np.sum(gnd)
        count = np.linspace(1, tsum, int(tsum))
        
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (np.squeeze(tindex)))
    map = map / len(query_pos_test)
    return map


def MAP(sess, model, test_i2t_pos, test_i2t, test_t2i_pos, test_t2i, feature_dict, label_dict):
    hash_dict_img = get_hash_dict(sess, model, feature_dict)
    
    map_i2t = calcu_map(test_i2t_pos, test_i2t, hash_dict_img, label_dict)
    map_t2i = calcu_map(test_t2i_pos, test_t2i, hash_dict_img, label_dict)
    return map_i2t, map_t2i

def get_hash_dict(sess, model, feature_dict):
    hash_dict = {}
    for item in feature_dict:
        input_data = np.asarray(feature_dict[item])
        input_data_dim = input_data.shape[0]
        input_data = input_data.reshape(1, input_data_dim)

        if item.split('.')[-1] == 'txt':
            output_hash = sess.run(model.text_hash, feed_dict={model.text_data: input_data})
        elif item.split('.')[-1] == 'jpg':
            output_hash = sess.run(model.image_hash, feed_dict={model.image_data: input_data})

        hash_dict[item] = output_hash[0]

    return hash_dict
