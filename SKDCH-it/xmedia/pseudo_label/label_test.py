# coding=UTF-8
import sys
sys.path.append('./')
from map import *
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import json

def get_AP(k_nearest, label, query_index, k):
    score = 0.0
    for i in range(k):
        if np.dot(label[query_index], label[int(k_nearest[i])]) > 0:
            score += 1.0
    return score / k

def get_knn(img_feat, txt_feat):
    train_img_label_xmedia = np.load('../../data_test/labels_train_img.npy', allow_pickle=True, encoding="latin1")
    train_txt_label_xmedia = np.load('../../data_test/labels_train_txt.npy', allow_pickle=True, encoding="latin1")
    semi_img_label_xmedia = np.load('../../data_test/labels_semi_img.npy', allow_pickle=True, encoding="latin1")
    semi_txt_label_xmedia = np.load('../../data_test/labels_semi_txt.npy', allow_pickle=True, encoding="latin1")

    retrieval_label_img = np.concatenate((train_img_label_xmedia, semi_img_label_xmedia), axis=0)
    retrieval_label_txt = np.concatenate((train_txt_label_xmedia, semi_txt_label_xmedia), axis=0)

    retrieval_size_img = 4000
    retrieval_size_txt = 4000
    K_img = 250
    K_txt = 250
    KNN_img = np.zeros((retrieval_size_img, K_img))
    KNN_txt = np.zeros((retrieval_size_txt, K_txt))
    accuracy_sum_img = 0
    accuracy_sum_txt = 0

    distance_img = pdist(img_feat, 'euclidean')
    distance_txt = pdist(txt_feat, 'euclidean')
    distance_img = squareform(distance_img)
    distance_txt = squareform(distance_txt)

    for j in range(retrieval_size_txt):
        k_nearest_txt = np.argsort(distance_txt[j])[0:K_txt]
        accuracy_sum_txt += get_AP(k_nearest_txt, retrieval_label_txt, j, K_txt)
        KNN_txt[j] = k_nearest_txt
    print(accuracy_sum_txt / retrieval_size_txt)

    for i in range(retrieval_size_img):
        k_nearest_img = np.argsort(distance_img[i])[0:K_img]
        accuracy_sum_img += get_AP(k_nearest_img, retrieval_label_img, i, K_img)
        KNN_img[i] = k_nearest_img
    print(accuracy_sum_img / retrieval_size_img)

    return KNN_img, KNN_txt

def extract_feature(sess, model, data, flag):
    num_data = len(data)
    batch_size = 256
    index = np.linspace(0, num_data - 1, num_data).astype(np.int32)

    feat_data = []
    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]

        data_batch = data[ind]
        if flag == 'image':
            output_feat = sess.run(model.image_sig, feed_dict={model.image_data: data_batch})
        elif flag == 'txt':
            output_feat = sess.run(model.text_sig, feed_dict={model.text_data: data_batch})
        feat_data.append(output_feat)
    feat_data = np.concatenate(feat_data)

    return feat_data

class update_labels():
    def __init__(self):
        self.num_train_txt = 4000
        self.num_train_img = 4000

        self.have_weight = True
        self.SEMANTIC_EMBED = 512
        self.MAX_ITER = 100

        self.batch_size = 256
        self.image_size = 224

        self.IMAGE_DIM = 4096
        self.TEXT_DIM = 3000

        self.OUTPUT_DIM = 16
        self.HIDDEN_DIM = 1024
        self.CLASS_DIM = 20
        self.WEIGHT_DECAY = 0.01
        self.D_LEARNING_RATE = 0.001
        self.G_LEARNING_RATE = 0.001
        self.TEMPERATURE = 0.2
        self.BETA = 5.0
        self.GAMMA = 0.1
        self.REWARD_BASE = 10.0
        self.WORKDIR_2 = '../../data_test/'
        self.WORKDIR = '../../data/'
        self.retrieval_txt_path_list = np.load(self.WORKDIR_2 + 'retrieval_txt_path_list_list.npy', allow_pickle=True,
                                          encoding="latin1")
        self.retrieval_img_path_list = np.load(self.WORKDIR_2 + 'retrieval_img_path_list_list.npy', allow_pickle=True,
                                          encoding="latin1")
        self.test_txt_path_list = np.load(self.WORKDIR_2 + 'test_txt_path_list_list.npy', allow_pickle=True, encoding="latin1")
        self.test_img_path_list = np.load(self.WORKDIR_2 + 'test_img_path_list_list.npy', allow_pickle=True, encoding="latin1")

        self.retrieval_txt = np.load(self.WORKDIR_2 + 'txt_retrieval.npy', allow_pickle=True, encoding="latin1")
        self.test_txt = np.load(self.WORKDIR_2 + 'txt_test.npy', allow_pickle=True, encoding="latin1")
        self.retrieval_img = np.load(self.WORKDIR_2 + 'img_retrieval.npy', allow_pickle=True, encoding="latin1")
        self.test_img = np.load(self.WORKDIR_2 + 'img_test.npy', allow_pickle=True, encoding="latin1")

        self.retrieval_x = self.retrieval_txt_path_list
        self.query_x = self.test_txt_path_list

        self.retrieval_y = self.retrieval_img
        self.query_y = self.test_img

        self.validation_x = np.load(self.WORKDIR_2 + 'database_txt_path_list_list.npy', allow_pickle=True, encoding="latin1")
        self.validation_y = np.load(self.WORKDIR_2 + 'img_database.npy', allow_pickle=True, encoding="latin1")
        self.validation_txt_feats = np.load(self.WORKDIR_2 + 'txt_database.npy', allow_pickle=True, encoding="latin1")
        self.test_txt_feats = self.test_txt

        self.validation_img_feats = np.load(self.WORKDIR_2 + 'img_database.npy', allow_pickle=True, encoding="latin1")
        self.test_img_feats = np.load(self.WORKDIR_2 + 'img_test.npy', allow_pickle=True, encoding="latin1")

        self.retrieval_txt_label = np.load(self.WORKDIR_2 + 'labels_retrieval_txt.npy', allow_pickle=True,
                                      encoding="latin1")
        self.retrieval_img_label = np.load(self.WORKDIR_2 + 'labels_retrieval_img.npy', allow_pickle=True, encoding="latin1")
        self.test_txt_label = np.load(self.WORKDIR_2 + 'labels_test_txt.npy', allow_pickle=True, encoding="latin1")
        self.test_img_label = np.load(self.WORKDIR_2 + 'labels_test_img.npy', allow_pickle=True, encoding="latin1")

        self.train_L_txt = np.load(self.WORKDIR_2 + 'labels_train_txt.npy', allow_pickle=True,
                              encoding="latin1")
        self.train_L_img = np.load(self.WORKDIR_2 + 'labels_train_img.npy', allow_pickle=True, encoding="latin1")

        self.validation_txt_label = np.load(self.WORKDIR_2 + 'labels_database_txt.npy', allow_pickle=True,
                                       encoding="latin1")
        self.validation_img_label = np.load(self.WORKDIR_2 + 'labels_database_img.npy', allow_pickle=True, encoding="latin1")
        self.num_train = 4000
        print('num_train', self.num_train)

    def updating(self, sess_g, discriminator):
        train_img = np.load('../../data_test/img_train.npy', allow_pickle=True, encoding="latin1")
        semi_img = np.load('../../data_test/img_semi.npy', allow_pickle=True, encoding="latin1")

        train_txt = np.load('../../data_test/txt_train.npy', allow_pickle=True, encoding="latin1")
        semi_txt = np.load('../../data_test/txt_semi.npy', allow_pickle=True, encoding="latin1")
        retrieval_img = np.concatenate((train_img, semi_img), axis=0)  # 特征
        retrieval_txt = np.concatenate((train_txt, semi_txt), axis=0)
        V_db = extract_feature(sess_g, discriminator, retrieval_img, 'image')
        T_db = extract_feature(sess_g, discriminator, retrieval_txt, 'txt')

        knn_img, knn_txt = get_knn(V_db, T_db)

        Sim_label_1 = (np.dot(self.train_L_txt, self.train_L_img.transpose()) > 0).astype(np.int32)
        Sim_label = Sim_label_1 * 0.999

        Sim_all = np.zeros((self.num_train, self.num_train))
        for i in range(self.num_train):
            ind = np.concatenate((knn_img[i], knn_txt[i])).astype(np.int32)
            Sim_all[i][ind] = 0.999

        Sim_all_row = np.vsplit(Sim_all, self.num_train)

        Sim_1 = []
        for i in range(self.train_L_txt.shape[0]):
            a = Sim_all_row[i]
            Sim_1.append(a[0])
        Sim_up_row = np.array(Sim_1)  # 4000 *4000

        Sim_right_row_1 = np.hsplit(Sim_up_row, 4000)

        Sim_3 = []
        for i in range(1000):
            c = Sim_right_row_1[i + 3000]
            Sim_3.append(c)
        Sim_right_row = np.concatenate(Sim_3, axis=1)  # 4000*1000

        Sim_2 = []
        for i in range(1000):
            b = Sim_all_row[i + 3000]
            Sim_2.append(b[0])
        Sim_down_row = np.array(Sim_2)  # 1000*4000

        Sim_4 = np.concatenate((Sim_label, Sim_right_row), axis=1)

        Sim = np.concatenate((Sim_4, Sim_down_row), axis=0)

        ans_dict = {}
        ans_max_dict = {}
        for i in range(Sim_down_row.shape[0]):
            ans_row = np.where(Sim_down_row[i] == 0.999)
            for j in range(len(ans_row[0])):
                score = 1
                if ans_row[0][j] < 3000:
                    Sim_down_label_1 = np.array(self.train_L_txt[ans_row[0][j]])
                    Sim_down_label = ",".join(str(i) for i in Sim_down_label_1)

                    if Sim_down_label not in ans_dict:
                        ans_dict[Sim_down_label] = score

                    else:
                        score = score + 1
                        ans_dict[Sim_down_label] = score

                    ans_max = max(ans_dict, key=ans_dict.get)
                    ans_max_dict[i] = ans_max
                ans_dict = {}

        ans_all = []
        for i in range(1000):
            ans = np.where(Sim_down_row[i] == 0.999)
            ans_all.append(ans)

        Sim_unlabel_row = Sim_down_row.shape[0]

        Sim_unlabel = np.zeros((Sim_unlabel_row, 20))
        for i in range(Sim_unlabel_row):
            one_str = ans_max_dict[i].split(',')
            one = []
            for k in one_str:
                k_list = json.loads(k)
                one.append(k_list)
            Sim_unlabel[i] = one

        train_L = np.concatenate((self.train_L_txt, Sim_unlabel), axis=0)

        np.save('../../data_test/T2I_train_64_semi.npy', Sim_unlabel)
        np.save('../../data_test/T2I_train_L.npy', train_L)

        Sim_label_1 = (np.dot(self.train_L_img, self.train_L_img.transpose()) > 0).astype(
            np.int32)
        Sim_label = Sim_label_1 * 0.999

        Sim_all = np.zeros((self.num_train, self.num_train))
        for i in range(self.num_train):
            ind = np.concatenate((knn_img[i], knn_txt[i])).astype(np.int32)
            Sim_all[i][ind] = 0.999

        Sim_all_row = np.vsplit(Sim_all, self.num_train)  # 按行分割

        Sim_1 = []
        for i in range(self.train_L_img.shape[0]):
            a = Sim_all_row[i]
            Sim_1.append(a[0])
        Sim_up_row = np.array(Sim_1)  # 4000 *4000

        Sim_right_row_1 = np.hsplit(Sim_up_row, 4000)

        Sim_3 = []
        for i in range(1000):
            c = Sim_right_row_1[i + 3000]
            Sim_3.append(c)
        Sim_right_row = np.concatenate(Sim_3, axis=1)  # 4000*1000

        Sim_2 = []
        for i in range(1000):
            b = Sim_all_row[i + 3000]
            Sim_2.append(b[0])
        Sim_down_row = np.array(Sim_2)  # 1000*4000
        Sim_4 = np.concatenate((Sim_label, Sim_right_row), axis=1)
        Sim = np.concatenate((Sim_4, Sim_down_row), axis=0)
        ans_dict = {}
        ans_max_dict = {}
        for i in range(Sim_down_row.shape[0]):
            ans_row = np.where(Sim_down_row[i] == 0.999)
            for j in range(len(ans_row[0])):
                score = 1
                if ans_row[0][j] < 3000:

                    Sim_down_label_1 = np.array(self.train_L_img[ans_row[0][j]])
                    Sim_down_label = ",".join(str(i) for i in Sim_down_label_1)

                    if Sim_down_label not in ans_dict:
                        ans_dict[Sim_down_label] = score

                    else:
                        score = score + 1
                        ans_dict[Sim_down_label] = score

                    ans_max = max(ans_dict, key=ans_dict.get)
                    ans_max_dict[i] = ans_max
                ans_dict = {}

        ans_all = []
        for i in range(1000):
            ans = np.where(Sim_down_row[i] == 0.999)
            ans_all.append(ans)

        Sim_unlabel_row = Sim_down_row.shape[0]

        Sim_unlabel = np.zeros((Sim_unlabel_row, 20))
        for i in range(Sim_unlabel_row):
            one_str = ans_max_dict[i].split(',')
            one = []
            for k in one_str:
                k_list = json.loads(k)
                one.append(k_list)
            Sim_unlabel[i] = one
        train_L = np.concatenate((self.train_L_img, Sim_unlabel), axis=0)

        np.save('../../data_test/I2T_train_64_semi.npy', Sim_unlabel)
        np.save('../../data_test/I2T_train_L.npy', train_L)
        return True