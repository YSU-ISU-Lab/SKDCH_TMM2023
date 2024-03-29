# -- coding: utf-8--
import sys

sys.path.append('./')
import pickle as cPickle
import os
import random
import tensorflow as tf
import utils as ut
from map import *
from dis_model_nn import DIS
from gen_model_nn import GEN
import tqdm

GPU_ID = '7'
OUTPUT_DIM = 64
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

SELECTNUM = 2
SAMPLERATIO = 20

WHOLE_EPOCH = 1000
D_EPOCH = 1
G_EPOCH = 2
GS_EPOCH = 30
D_DISPLAY = 1
G_DISPLAY = 2

TEXT_DIM = 3000
IMAGE_DIM = 4096
HIDDEN_DIM = 8192
CLASS_DIM = 20
BATCH_SIZE = 256
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.0001
G_LEARNING_RATE = 0.01
BETA = 5
GAMMA = 0.1
REWARD_BASE = 10.0

WORKDIR_2 = '../../dataset/'
DIS_MODEL_PRETRAIN_FILE = '../SCHGAN_I2T/data/I2T_student_best_pretrain_' + str(OUTPUT_DIM) + '.model'
DIS_MODEL_BEST_FILE = '../SCHGAN_I2T/data/I2T_student_' + str(OUTPUT_DIM) + '.model'
DIS_MODEL_NEWEST_FILE = '../SCHGAN_I2T/data/I2T_student_latest_' + str(OUTPUT_DIM) + '.model'
GEN_MODEL_BEST_FILE = '../SCHGAN_I2T/data/I2T_student_' + str(OUTPUT_DIM) + '.model'
GEN_MODEL_NEWEST_FILE = '../SCHGAN_I2T/data/I2T_student_latest_' + str(OUTPUT_DIM) + '.model'

train_t2i, train_t2i_pos, train_t2i_neg, train_t2i_pair, train_i2t, train_i2t_pos, train_i2t_neg, train_i2t_pair, test_t2i, test_t2i_pos, test_i2t, test_i2t_pos = ut.load_all_query_url(
    WORKDIR_2, CLASS_DIM)
feature_dict = ut.load_all_feature_for_train_2(WORKDIR_2)
feature_test_dict = ut.load_all_feature_for_train_test_2(WORKDIR_2)
label_dict = ut.load_all_label_2(WORKDIR_2)

def generate_samples(train_pos, train_neg):
    data = []
    for query in train_pos:
        pos_list = train_pos[query]
        candidate_neg_list = train_neg[query]
        
        random.shuffle(pos_list)
        random.shuffle(candidate_neg_list)
        
        neg_list = []
        for i in range(SELECTNUM):
            neg_list.append(candidate_neg_list[i])
        
        for i in range(SELECTNUM):
            data.append((query, pos_list[i], neg_list[i]))
    
    random.shuffle(data)
    return data


def train_discriminator(sess, discriminator, dis_train_list, query_pair_train, flag):
    train_size = len(dis_train_list)
    index = 1
    while index < train_size:
        input_query = []
        input_pos = []
        input_neg = []
        pos_pair_label = []
        neg_pair_label = []
        
        if index + BATCH_SIZE <= train_size:
            for i in range(index, index + BATCH_SIZE):
                query, pos, neg = dis_train_list[i]
                input_query.append(feature_dict[query])
                input_pos.append(feature_dict[pos])
                input_neg.append(feature_dict[neg])
                
                if query_pair_train[query] == pos:
                    pos_pair_label.append(1.0)
                else:
                    pos_pair_label.append(0.0)
                if query_pair_train[query] == neg:
                    neg_pair_label.append(1.0)
                else:
                    neg_pair_label.append(0.0)
        else:
            for i in range(index, train_size):
                query, pos, neg = dis_train_list[i]
                input_query.append(feature_dict[query])
                input_pos.append(feature_dict[pos])
                input_neg.append(feature_dict[neg])
                
                if query_pair_train[query] == pos:
                    pos_pair_label.append(1.0)
                else:
                    pos_pair_label.append(0.0)
                if query_pair_train[query] == neg:
                    neg_pair_label.append(1.0)
                else:
                    neg_pair_label.append(0.0)
        
        index += BATCH_SIZE
        
        query_data = np.asarray(input_query)
        input_pos = np.asarray(input_pos)
        input_neg = np.asarray(input_neg)
        pos_pair_label = np.asarray(pos_pair_label)
        neg_pair_label = np.asarray(neg_pair_label)
        
        if flag == 't2i':
            d_loss = sess.run(discriminator.t2i_loss,
                              feed_dict={discriminator.text_data: query_data,
                                         discriminator.image_data: input_pos,
                                         discriminator.image_neg_data: input_neg,
                                         discriminator.pos_pair_label: pos_pair_label,
                                         discriminator.neg_pair_label: neg_pair_label})
            _ = sess.run(discriminator.t2i_updates,
                         feed_dict={discriminator.text_data: query_data,
                                    discriminator.image_data: input_pos,
                                    discriminator.image_neg_data: input_neg,
                                    discriminator.pos_pair_label: pos_pair_label,
                                    discriminator.neg_pair_label: neg_pair_label})
        elif flag == 'i2t':
            d_loss = sess.run(discriminator.i2t_loss,
                              feed_dict={discriminator.image_data: query_data,
                                         discriminator.text_data: input_pos,
                                         discriminator.text_neg_data: input_neg,
                                         discriminator.pos_pair_label: pos_pair_label,
                                         discriminator.neg_pair_label: neg_pair_label})
            _ = sess.run(discriminator.i2t_updates,
                         feed_dict={discriminator.image_data: query_data,
                                    discriminator.text_data: input_pos,
                                    discriminator.text_neg_data: input_neg,
                                    discriminator.pos_pair_label: pos_pair_label,
                                    discriminator.neg_pair_label: neg_pair_label})
    
    print('D_Loss_%s: %.4f' % (flag, d_loss))
    return discriminator

def train_generator(sess, generator, discriminator, train_neg, train_pos, flag):
    for query in train_pos.keys():
        pos_list = train_pos[query]
        candidate_list = train_neg[query]

        random.shuffle(candidate_list)
        sample_size = int(len(candidate_list) / SAMPLERATIO)
        candidate_list = candidate_list[0: sample_size]

        random.shuffle(pos_list)
        pos_list = pos_list[0:SELECTNUM]

        candidate_data = np.asarray([feature_dict[url] for url in candidate_list])
        if flag == 't2i':
            query_data = np.asarray(feature_dict[query]).reshape(1, TEXT_DIM)
            candidate_score = sess.run(generator.pred_score,
                                       feed_dict={generator.text_data: query_data,
                                                  generator.image_data: candidate_data})
        elif flag == 'i2t':
            query_data = np.asarray(feature_dict[query]).reshape(1, IMAGE_DIM)
            candidate_score = sess.run(generator.pred_score,
                                       feed_dict={generator.text_data: candidate_data,
                                                  generator.image_data: query_data})

        exp_rating = np.exp(candidate_score)
        prob = exp_rating / np.sum(exp_rating)
        neg_index = np.random.choice(np.arange(len(candidate_list)), size=[SELECTNUM], p=prob)
        neg_list = np.array(candidate_list)[neg_index]
        neg_index = np.asarray(neg_index)

        input_pos = np.asarray([feature_dict[url] for url in pos_list])
        input_neg = np.asarray([feature_dict[url] for url in neg_list])
        query_label = np.asarray(label_dict[query]).reshape(1, CLASS_DIM)
        neg_label = np.asarray([label_dict[url] for url in neg_list])

        if flag == 't2i':
            neg_reward = sess.run(discriminator.t2i_reward,
                                  feed_dict={discriminator.text_data: query_data,
                                             discriminator.image_data: input_pos,
                                             discriminator.image_neg_data: input_neg})
            g_loss = sess.run(generator.gen_loss,
                              feed_dict={generator.text_data: query_data,
                                         generator.text_label: query_label,
                                         generator.image_data: input_neg,
                                         generator.image_label: neg_label,
                                         generator.reward: neg_reward})
            _ = sess.run(generator.gen_updates,
                         feed_dict={generator.text_data: query_data,
                                    generator.text_label: query_label,
                                    generator.image_data: input_neg,
                                    generator.image_label: neg_label,
                                    generator.reward: neg_reward})

        elif flag == 'i2t':
            neg_reward = sess.run(discriminator.i2t_reward,
                                  feed_dict={discriminator.image_data: query_data,
                                             discriminator.text_data: input_pos,
                                             discriminator.text_neg_data: input_neg})
            g_loss = sess.run(generator.gen_loss,
                              feed_dict={generator.image_data: query_data,
                                         generator.image_label: query_label,
                                         generator.text_data: input_neg,
                                         generator.text_label: neg_label,
                                         generator.reward: neg_reward})
            _ = sess.run(generator.gen_updates,
                         feed_dict={generator.image_data: query_data,
                                    generator.image_label: query_label,
                                    generator.text_data: input_neg,
                                    generator.text_label: neg_label,
                                    generator.reward: neg_reward})

    print('G_Loss_%s: %.4f' % (flag, g_loss))
    return generator

def main():
    with tf.device('/gpu:' + GPU_ID):
        dis_param = cPickle.load(open(DIS_MODEL_PRETRAIN_FILE))
        discriminator = DIS(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA,
                            REWARD_BASE, param=dis_param)
        generator = GEN(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, CLASS_DIM, WEIGHT_DECAY, G_LEARNING_RATE,
                        param=None)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.initialize_all_variables())

        map_best_val_gen = 0.0
        map_best_val_dis = 0.0
        
        for epoch in tqdm.tqdm(range(WHOLE_EPOCH), desc='epoch'):
            for g_epoch in range(D_EPOCH):
                print('g_epoch: ' + str(g_epoch))
                generator = train_generator(sess, generator, discriminator, train_t2i_neg, train_t2i_pos, 't2i')
                generator = train_generator(sess, generator, discriminator, train_i2t_neg, train_i2t_pos, 'i2t')

                if (g_epoch + 1) % (G_DISPLAY) == 0:
                    i2t_test_map, t2i_test_map = MAP(sess, generator, test_i2t_pos, test_i2t, test_t2i_pos,
                                                     test_t2i, feature_dict, label_dict)
                    print('---------------------------------------------------------------')
                    print('pretrain_I2T_Test_MAP: %.4f' % i2t_test_map)
                    print('pretrain_T2I_Test_MAP: %.4f' % t2i_test_map)
                    print('---------------------------------------------------------------')
                    average_map = 0.5 * (i2t_test_map + t2i_test_map)
                    if average_map > map_best_val_gen:
                        map_best_val_gen = average_map
                        generator.save_model(sess, GEN_MODEL_BEST_FILE)
                generator.save_model(sess, GEN_MODEL_NEWEST_FILE)

            for d_epoch in range(D_EPOCH):
                if d_epoch % GS_EPOCH == 0:
                    print('negative image sampling for d using g ...')
                    dis_train_t2i_list = generate_samples(train_t2i_pos, train_t2i_neg)
                    print('negative text sampling for d using g ...')
                    dis_train_i2t_list = generate_samples(train_i2t_pos, train_i2t_neg)
                
                discriminator = train_discriminator(sess, discriminator, dis_train_t2i_list, train_t2i_pair, 't2i')
                discriminator = train_discriminator(sess, discriminator, dis_train_i2t_list, train_i2t_pair, 'i2t')
                if (d_epoch + 1) % (D_DISPLAY) == 0:
                    i2t_test_map, t2i_test_map = MAP(sess, discriminator, test_i2t_pos, test_i2t, test_t2i_pos,
                                                     test_t2i, feature_dict, label_dict)
                    print('---------------------------------------------------------------')
                    print('train_i2t_Test_MAP: %.4f' % i2t_test_map)
                    print('train_t2i_Test_MAP: %.4f' % t2i_test_map)
                    print('---------------------------------------------------------------')
                    average_map = 0.5 * (i2t_test_map + t2i_test_map)
                    if average_map > map_best_val_dis:
                        map_best_val_dis = average_map
                        discriminator.save_model(sess, DIS_MODEL_BEST_FILE)
                discriminator.save_model(sess, DIS_MODEL_NEWEST_FILE)
        
        sess.close()


if __name__ == '__main__':
    main()

