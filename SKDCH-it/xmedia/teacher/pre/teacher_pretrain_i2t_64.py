# coding=UTF-8
import sys
sys.path.append('./')
import os
import random
import tensorflow as tf
import utils as ut
from map_pretrain import *
from dis_model_nn import DIS
import tqdm

GPU_ID = '9'
OUTPUT_DIM = 64

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
SELECTNUM = 4
SAMPLERATIO = 20

WHOLE_EPOCH = 1
D_EPOCH = 1000
GS_EPOCH = 1000
D_DISPLAY = 1

TEXT_DIM = 3000
IMAGE_DIM = 4096
HIDDEN_DIM = 8192
CLASS_DIM = 20
BATCH_SIZE = 256
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.01
BETA = 5
GAMMA = 0.1

WORKDIR = '../SCHGAN_I2T/xmedia/'
WORKDIR_2 = '../SCHGAN_I2T/data/'
DIS_MODEL_PRETRAIN_FILE = WORKDIR_2 + 'I2T_teacher_best_pretrain_' + str(OUTPUT_DIM) + '.model'
DIS_MODEL_PRETRAIN_NEWEST_FILE = WORKDIR_2 + 'I2T_teacher_newest_pretrain_' + str(OUTPUT_DIM) + '.model'

train_t2i, train_t2i_pos, train_t2i_neg, train_t2i_pair, train_i2t, train_i2t_pos, train_i2t_neg, train_i2t_pair, test_t2i, test_t2i_pos, test_i2t, test_i2t_pos = ut.load_all_query_url(WORKDIR + 'list/', CLASS_DIM)

feature_dict = ut.load_all_feature(WORKDIR + 'list/', WORKDIR + 'feature_znorm/')
label_dict = ut.load_all_label(WORKDIR + 'list/')

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
			_ = sess.run(discriminator.t2i_updates,
						 feed_dict={discriminator.text_data: query_data,
									discriminator.image_data: input_pos,
									discriminator.image_neg_data: input_neg,
									discriminator.pos_pair_label: pos_pair_label,
									discriminator.neg_pair_label: neg_pair_label})
			d_loss = sess.run(discriminator.t2i_loss,
						 feed_dict={discriminator.text_data: query_data,
									discriminator.image_data: input_pos,
									discriminator.image_neg_data: input_neg,
									discriminator.pos_pair_label: pos_pair_label,
									discriminator.neg_pair_label: neg_pair_label})
		elif flag == 'i2t':
			_ = sess.run(discriminator.i2t_updates,
						 feed_dict={discriminator.image_data: query_data,
									discriminator.text_data: input_pos,
									discriminator.text_neg_data: input_neg,
									discriminator.pos_pair_label: pos_pair_label,
									discriminator.neg_pair_label: neg_pair_label})
			d_loss = sess.run(discriminator.i2t_loss,
						 feed_dict={discriminator.image_data: query_data,
									discriminator.text_data: input_pos,
									discriminator.text_neg_data: input_neg,
									discriminator.pos_pair_label: pos_pair_label,
									discriminator.neg_pair_label: neg_pair_label})
	
	print('D_Loss_%s: %.4f' % (flag,d_loss))
	return discriminator
	
def main():
	with tf.device('/gpu:' + GPU_ID):
		discriminator = DIS(IMAGE_DIM, TEXT_DIM,HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA, loss = 'svm', param = None)
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		sess.run(tf.initialize_all_variables())

		print('start adversarial training')
		map_best_val_dis = 0.0

		for epoch in tqdm.tqdm(range(WHOLE_EPOCH), desc='epoch'):
			print('Training D ...')
			for d_epoch in range(D_EPOCH):
				print('d_epoch: ' + str(d_epoch))
				if d_epoch % GS_EPOCH == 0:
					print('negative image sampling for d using g ...')
					dis_train_t2i_list = generate_samples(train_t2i_pos, train_t2i_neg)
					print('negative text sampling for d using g ...')
					dis_train_i2t_list = generate_samples(train_i2t_pos, train_i2t_neg)
				
				discriminator = train_discriminator(sess, discriminator, dis_train_t2i_list, train_t2i_pair, 't2i')
				discriminator = train_discriminator(sess, discriminator, dis_train_i2t_list, train_i2t_pair, 'i2t')
				
				if (d_epoch + 1) % (D_DISPLAY) == 0:
					i2t_test_map, t2i_test_map = MAP(sess, discriminator, test_i2t_pos, test_i2t, test_t2i_pos,test_t2i, feature_dict, label_dict)
					print('---------------------------------------------------------------')
					print('train_i2t_Test_MAP: %.4f' % i2t_test_map)
					print('train_t2i_Test_MAP: %.4f' % t2i_test_map)
					print('---------------------------------------------------------------')
					with open('record_' + str(OUTPUT_DIM) + '.txt', 'a') as record_file:
						record_file.write('t2i_Test_MAP_32: %.4f\n' % t2i_test_map)
						record_file.write('i2t_Test_MAP_32: %.4f\n' % i2t_test_map)
					
					average_map = 0.5 * (t2i_test_map + i2t_test_map)
					if average_map > map_best_val_dis:
						map_best_val_dis = average_map
						discriminator.save_model(sess, DIS_MODEL_PRETRAIN_FILE)
				discriminator.save_model(sess, DIS_MODEL_PRETRAIN_NEWEST_FILE)

		sess.close()
if __name__ == '__main__':
	main()
