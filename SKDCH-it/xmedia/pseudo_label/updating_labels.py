from dis_model_nn import DIS
import label_test as gl
import pickle as cPickle
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

IMAGE_DIM = 4096
TEXT_DIM = 3000
OUTPUT_DIM = 64
HIDDEN_DIM = 8192
CLASS_DIM = 20
BATCH_SIZE = 256
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.0001
G_LEARNING_RATE = 0.01
BETA = 5
GAMMA = 0.1
REWARD_BASE = 10.0

WORKDIR_2 = ''
DIS_MODEL_BEST_FILE = '../../data/I2T_teacher_64.model'

dis_param = cPickle.load(open(DIS_MODEL_BEST_FILE, 'rb'), encoding='iso-8859-1')
discriminator = DIS(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA,
                            REWARD_BASE, param=dis_param)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())

up = gl.update_labels()
if up.updating(sess, discriminator):
    print('success')
else:
    print('error')