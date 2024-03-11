# -- coding: utf-8--
import tensorflow as tf
import pickle as cPickle


class DIS():
    def __init__(self, image_dim, text_dim, hidden_dim, output_dim, weight_decay, learning_rate, beta, gamma,reward_base, loss='svm', param=None):
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.params = []
        
        self.text_data = tf.placeholder(tf.float32, shape=[None, self.text_dim], name="text_data")
        self.image_data = tf.placeholder(tf.float32, shape=[None, self.image_dim], name="image_data")
        self.text_pos_data = tf.placeholder(tf.float32, shape=[None, self.text_dim], name="text_pos_data")
        self.image_pos_data = tf.placeholder(tf.float32, shape=[None, self.image_dim], name="image_pos_data")
        
        self.text_neg_data = tf.placeholder(tf.float32, shape=[None, self.text_dim], name="text_neg_data")
        self.image_neg_data = tf.placeholder(tf.float32, shape=[None, self.image_dim], name="image_neg_data")
        self.pos_pair_label = tf.placeholder(tf.float32, shape=[None], name="pos_pair_label")
        self.neg_pair_label = tf.placeholder(tf.float32, shape=[None], name="neg_pair_label")
        
        with tf.variable_scope('discriminator'):
            if param == None:
                self.Wq_1 = tf.get_variable('Wq_1', [self.image_dim, self.hidden_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Wq_2 = tf.get_variable('Wq_2', [self.hidden_dim, self.output_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Bq_1 = tf.get_variable('Bq_1', [self.hidden_dim], initializer=tf.constant_initializer(0.0))
                self.Bq_2 = tf.get_variable('Bq_2', [self.output_dim], initializer=tf.constant_initializer(0.0))
                
                self.Wc_1 = tf.get_variable('Wc_1', [self.text_dim, self.hidden_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Wc_2 = tf.get_variable('Wc_2', [self.hidden_dim, self.output_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Bc_1 = tf.get_variable('Bc_1', [self.hidden_dim], initializer=tf.constant_initializer(0.0))
                self.Bc_2 = tf.get_variable('Bc_2', [self.output_dim], initializer=tf.constant_initializer(0.0))
            else:
                self.Wq_1 = tf.Variable(param[0])
                self.Wq_2 = tf.Variable(param[1])
                self.Bq_1 = tf.Variable(param[2])
                self.Bq_2 = tf.Variable(param[3])
                
                self.Wc_1 = tf.Variable(param[4])
                self.Wc_2 = tf.Variable(param[5])
                self.Bc_1 = tf.Variable(param[6])
                self.Bc_2 = tf.Variable(param[7])
            
            self.params.append(self.Wq_1)
            self.params.append(self.Wq_2)
            self.params.append(self.Bq_1)
            self.params.append(self.Bq_2)
            
            self.params.append(self.Wc_1)
            self.params.append(self.Wc_2)
            self.params.append(self.Bc_1)
            self.params.append(self.Bc_2)

        self.image_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.image_data, self.Wq_1, self.Bq_1)), self.Wq_2, self.Bq_2)
        self.image_sig = tf.sigmoid(self.image_rep)
        self.image_hash = tf.cast(self.image_sig + 0.5, tf.int32)
        
        self.text_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.text_data, self.Wc_1, self.Bc_1)), self.Wc_2, self.Bc_2)
        self.text_sig = tf.sigmoid(self.text_rep)
        self.text_hash = tf.cast(self.text_sig + 0.5, tf.int32)
        
        self.image_pos_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.image_pos_data, self.Wq_1, self.Bq_1)), self.Wq_2, self.Bq_2)
        self.image_pos_sig = tf.sigmoid(self.image_pos_rep)
        
        self.text_pos_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.text_pos_data, self.Wc_1, self.Bc_1)), self.Wc_2, self.Bc_2)
        self.text_pos_sig = tf.sigmoid(self.text_pos_rep)

        self.pred_distance = tf.reduce_sum(tf.square(self.text_sig - self.image_sig), 1)
        self.pred_img_pos_distance = tf.reduce_sum(tf.square(self.image_sig - self.image_pos_sig), 1)
        self.pred_txt_pos_distance = tf.reduce_sum(tf.square(self.text_sig - self.text_pos_sig), 1)
        
        self.hash_score = tf.reduce_sum(tf.cast(tf.equal(self.text_hash, self.image_hash), tf.float32), 1)

        self.image_neg_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.image_neg_data, self.Wq_1, self.Bq_1)), self.Wq_2, self.Bq_2)
        self.image_neg_sig = tf.sigmoid(self.image_neg_rep)
        
        self.text_neg_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.text_neg_data, self.Wc_1, self.Bc_1)), self.Wc_2, self.Bc_2)
        self.text_neg_sig = tf.sigmoid(self.text_neg_rep)
        
        self.pred_t2i_neg_distance = tf.reduce_sum(tf.square(self.text_sig - self.image_neg_sig), 1)
        self.pred_i2t_neg_distance = tf.reduce_sum(tf.square(self.image_sig - self.text_neg_sig), 1)
        self.pred_t2t_neg_distance = tf.reduce_sum(tf.square(self.text_sig - self.text_neg_sig), 1)
        self.pred_v2v_neg_distance = tf.reduce_sum(tf.square(self.image_sig - self.image_neg_sig), 1)
        
        if loss == 'svm':
            with tf.name_scope('svm_loss'):
                self.t2i_loss = tf.reduce_mean(
                    tf.maximum(0.0, beta + self.pred_distance - 0.5 * (self.pred_t2i_neg_distance + self.pred_v2v_neg_distance)) + 0.02 * tf.maximum(0.0, 0.1 + self.pred_distance)) + \
                                gamma * tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=-self.pred_distance, labels=self.pos_pair_label)) + \
                                gamma * tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=-self.pred_t2i_neg_distance,
                                                            labels=self.neg_pair_label)) + \
                                self.weight_decay * (tf.nn.l2_loss(self.Wq_1) + tf.nn.l2_loss(self.Wq_2)
                                                     + tf.nn.l2_loss(self.Bq_1) + tf.nn.l2_loss(self.Bq_2)
                                                     + tf.nn.l2_loss(self.Wc_1) + tf.nn.l2_loss(self.Wc_2)
                                                     + tf.nn.l2_loss(self.Bc_1) + tf.nn.l2_loss(self.Bc_2))
                self.t2i_reward = tf.pow(reward_base, -tf.abs(
                    beta + self.pred_distance - self.pred_t2i_neg_distance))

                self.i2t_loss = tf.reduce_mean(
                    tf.maximum(0.0, beta + self.pred_distance - 0.5 * (self.pred_i2t_neg_distance  + self.pred_t2t_neg_distance)) + 0.02 * tf.maximum(0.0, 0.1 + self.pred_distance)) + \
                                gamma * tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=-self.pred_distance, labels=self.pos_pair_label)) + \
                                gamma * tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=-self.pred_i2t_neg_distance,
                                                            labels=self.neg_pair_label)) + \
                                self.weight_decay * (tf.nn.l2_loss(self.Wq_1) + tf.nn.l2_loss(self.Wq_2)
                                                     + tf.nn.l2_loss(self.Bq_1) + tf.nn.l2_loss(self.Bq_2)
                                                     + tf.nn.l2_loss(self.Wc_1) + tf.nn.l2_loss(self.Wc_2)
                                                     + tf.nn.l2_loss(self.Bc_1) + tf.nn.l2_loss(self.Bc_2))
                self.i2t_reward = tf.pow(reward_base, -tf.abs(beta + self.pred_distance - self.pred_i2t_neg_distance))
                self.t2t_loss = tf.reduce_mean(
                    tf.maximum(0.0, beta + self.pred_txt_pos_distance - self.pred_t2t_neg_distance)) + \
                                gamma * tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=-self.pred_txt_pos_distance,
                                                            labels=self.pos_pair_label)) + \
                                gamma * tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=-self.pred_t2t_neg_distance,
                                                            labels=self.neg_pair_label)) + \
                                self.weight_decay * (tf.nn.l2_loss(self.Wq_1) + tf.nn.l2_loss(self.Wq_2)
                                                     + tf.nn.l2_loss(self.Bq_1) + tf.nn.l2_loss(self.Bq_2))

        # 学习率指数下降
        self.global_step = tf.Variable(0, trainable=False)
        self.lr_step = tf.train.exponential_decay(self.learning_rate, self.global_step, 20000, 0.96,
                                                  staircase=True)  ##dis每2000步衰减一次
        self.t2i_optimizer = tf.train.GradientDescentOptimizer(self.lr_step)
        self.t2i_updates = self.t2i_optimizer.minimize(self.t2i_loss, var_list=self.params)
        
        self.i2t_optimizer = tf.train.GradientDescentOptimizer(self.lr_step)
        self.i2t_updates = self.i2t_optimizer.minimize(self.i2t_loss, var_list=self.params)
    
    def save_model(self, sess, filename):
        param = sess.run(self.params)
        cPickle.dump(param, open(filename, 'wb'))
