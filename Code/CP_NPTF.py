

import utils_funcs
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import joblib as jb

import sys
#run as
print("usage : python *.py gpu=0 rank=5 dataset=article lr=0.001")

print('start')
print( sys.argv)
#parse args
py_name = sys.argv[0]

args = sys.argv[1:]
args_dict  = {}
for arg_pair in args:
    arg, val_str = arg_pair.split( '=')
    args_dict[ arg] = val_str

arg_gpu_idx_str = args_dict['gpu']
arg_rank = int( args_dict['rank'])
arg_data_name = args_dict['dataset']
arg_lr = float( args_dict['lr'])

print( 'gpu index = %s' % arg_gpu_idx_str)
print( 'rank = %d' % arg_rank )
print( 'learning rate = %e' % arg_lr)



np.random.seed(47)
tf.set_random_seed(47)

from utils_funcs import  FLOAT_TYPE, NN_MAX, MATRIX_JITTER


class NTF_HP:
    def __init__(self, train_ind, train_y, init_config):
        self.train_ind = train_ind
        self.train_y = train_y
        self.nmod = train_ind.shape[1]
        self.uniq_ind, self.n_i, self.sq_sum, self.log_sum = utils_funcs.extract_event_tensor_Reileigh( self.train_ind, self.train_y)

        self.num_entries = len(self.uniq_ind)

        #self.log_file_name = init_config['log_file_name']
        self.init_U = init_config['U']
        self.batch_size_entry = init_config['batch_size_entry']
        self.learning_rate = init_config['learning_rate']

        self.GP_SCOPE_NAME = "gp_params"
        #GP parameters
        with tf.variable_scope( self.GP_SCOPE_NAME):
            # Embedding params
            self.tf_U = [tf.Variable(self.init_U[k], dtype=FLOAT_TYPE) for k in range(self.nmod)]

        # Integral Term
        # sum_i < int_0^T lam_i>
        # placeholders
        self.batch_entry_ind = tf.placeholder(dtype=tf.int32, shape=[self.batch_size_entry, self.nmod])
        self.batch_entry_n_i = tf.placeholder( dtype=FLOAT_TYPE, shape=[ self.batch_size_entry, 1])
        self.batch_entry_log_sum = tf.placeholder( dtype=FLOAT_TYPE, shape = [ self.batch_size_entry, 1])
        self.batch_entry_sq_sum = tf.placeholder( dtype= FLOAT_TYPE, shape=[ self.batch_size_entry,1])

        self.tf_T = tf.constant(self.train_y[-1][0] - self.train_y[0][0], dtype=FLOAT_TYPE)
        self.tf_T0 = tf.constant( self.train_y[0][0], dtype=FLOAT_TYPE)
        self.tf_T1 = tf.constant( self.train_y[-1][0], dtype=FLOAT_TYPE)

        # sample posterior base rate ( f )
        self.gp_base_rate_entries = utils_funcs.log_CP_base_rate( self.tf_U, self.batch_entry_ind)
        self.base_rate_entries = tf.exp( self.gp_base_rate_entries)

        #int term 1, using entryEvent
        self.int_part1 = self.num_entries / self.batch_size_entry  * tf.reduce_sum( self.base_rate_entries * self.batch_entry_sq_sum)

        # event sum term 1
        self.eventSum = ( self.batch_entry_n_i * self.gp_base_rate_entries + self.batch_entry_log_sum)
        self.event_sum_part1 = self.num_entries / self.batch_size_entry * ( tf.reduce_sum( self.eventSum))

        self.ELBO = self.event_sum_part1 - self.int_part1
        self.neg_ELBO = - self.ELBO
        self.ELBO_hist = []

        # setting
        self.min_opt = tf.train.AdamOptimizer(self.learning_rate)
        self.min_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.GP_SCOPE_NAME)
        #print( self.min_params) ##
        self.min_step = self.min_opt.minimize(self.neg_ELBO, var_list=self.min_params)

        # GPU settings
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.entries_ind_y_gnrt = utils_funcs.DataGenerator(self.uniq_ind, np.concatenate( [ self.n_i, self.sq_sum, self.log_sum], axis=1))
        self.isTestGraphInitialized = False


    def train(self, steps = 1, print_every = 100, test_every = False,
              val_error = False, val_all_ind = None, val_all_y = None,
              val_test_ind = None, val_test_y = None, verbose = False):
        print('start')
        for step in range( 1, steps + 1):
            if step % print_every == 0:
                print( "step = %d " %  step )

            # max step=============>>
            batch_entries_ind, batch_entries_info = self.entries_ind_y_gnrt.draw_next( self.batch_size_entry)
            batch_n_i = batch_entries_info[:,0:1]
            batch_sq_sum = batch_entries_info[:, 1:2]
            batch_log_sum = batch_entries_info[:, 2:3]


            train_feed_dict = { self.batch_entry_ind : batch_entries_ind,
                                self.batch_entry_n_i : batch_n_i,
                                self.batch_entry_sq_sum : batch_sq_sum,
                                self.batch_entry_log_sum: batch_log_sum}


            _, ELBO_ret_min_step, int_part1, sum_part1,  gp_base_rate_entries = self.sess.run(
                [self.min_step, self.neg_ELBO, self.int_part1, self.event_sum_part1,  self.gp_base_rate_entries], feed_dict=train_feed_dict)
            self.ELBO_hist.append( ELBO_ret_min_step)


            if step % print_every == 0:
                print("neg ELBO min step = %g, int_part1 = %g, sum_part1 = %g, max gp = %g, min gp = %g"
                      % ( ELBO_ret_min_step, int_part1, sum_part1, np.max( gp_base_rate_entries), np.min( gp_base_rate_entries) ))
            #<<===================End max step
        return self

    def create_standAlone_test_graph(self, test_ind, test_y):
        print("Create testing graph")
        self.test_ind = test_ind
        self.test_y = test_y

        self.num_test_events = len(test_ind)
        self.uniq_ind_test, self.n_i_test, self.sq_sum_test, self.log_sum_test  = utils_funcs.extract_event_tensor_Reileigh( self.test_ind, self.test_y)
        self.num_uniq_ind_test = len(self.uniq_ind_test)

        # Integral Term
        # sum_i < int_0^T lam_i>
        # placeholders
        self.entry_ind_test = tf.constant( self.uniq_ind_test, dtype=tf.int32 )
        self.entry_n_i_test = tf.constant( self.n_i_test, dtype=FLOAT_TYPE)
        self.entry_sq_sum_test = tf.constant( self.sq_sum_test, dtype=FLOAT_TYPE)
        self.entry_log_sum_test = tf.constant( self.log_sum_test, dtype=FLOAT_TYPE)

        self.gp_base_rate_entries_test = utils_funcs.log_CP_base_rate( self.tf_U, self.entry_ind_test)
        self.base_rate_entries_test = tf.exp(self.gp_base_rate_entries_test)

        # int term 1, using entryEvent
        self.int_part_test =  tf.reduce_sum(self.base_rate_entries_test * self.entry_sq_sum_test)


        # event sum term 1
        self.event_sum_test = tf.reduce_sum(self.gp_base_rate_entries_test * self.entry_n_i_test + self.entry_log_sum_test)
        self.llk_test = self.event_sum_test - self.int_part_test

        self.isTestGraphInitialized = True

        return self

    def test(self, verbose = False):
        if not self.isTestGraphInitialized:
            raise NameError("Test Graph hasn't been initialized")

        int_term, eventsum_term, test_llk = self.sess.run( [ self.int_part_test, self.event_sum_test, self.llk_test])

        return test_llk, int_term , eventsum_term


    def check_vars(self, var_list):
        batch_entries_ind, batch_entries_info = self.entries_ind_y_gnrt.draw_last()
        batch_n_i = batch_entries_info[:, 0:1]
        batch_sq_sum = batch_entries_info[:, 1:2]
        batch_log_sum = batch_entries_info[:, 2:3]

        train_feed_dict = {self.batch_entry_ind: batch_entries_ind,
                           self.batch_entry_n_i: batch_n_i,
                           self.batch_entry_sq_sum: batch_sq_sum,
                           self.batch_entry_log_sum: batch_log_sum}

        ret = self.sess.run( var_list, feed_dict=train_feed_dict)
        return ret


def test_data_set():
    (ind, y), (train_ind, train_y), (test_ind, test_y) = utils_funcs.load_dataSet(arg_data_name, '../Data')

    nmod = ind.shape[1]
    nvec = np.max(ind, axis=0) + 1

    R = arg_rank

    U = [np.random.rand(nvec[k], R) for k in range(nmod)]

    init_config = {}
    init_config['U'] = U
    init_config['batch_size_event'] = 64
    init_config['batch_size_entry'] = 64

    init_config['learning_rate'] = arg_lr

    model = NTF_HP(train_ind, train_y, init_config)
    model.create_standAlone_test_graph(test_ind, test_y)
    steps_per_epoch = int(len(train_ind) / init_config['batch_size_event'])
    num_epoch = 50

    test_llk = []
    for epoch in range(1, num_epoch + 1):
        print('epoch %d' % epoch)
        model.train(steps_per_epoch, int(steps_per_epoch))
        test_log_p, int_term, eventSum_term = model.test(verbose=False)
        print("test_log_llk = %g, int_term = %g,  eventsum_term = %g\n" % (test_log_p, int_term, eventSum_term))
        test_llk.append(test_log_p)

    utils_funcs.log_results('CP_NPTF.txt', arg_data_name, arg_rank, arg_lr, test_llk)
    model.sess.close()

if __name__ == '__main__':
    test_data_set()











