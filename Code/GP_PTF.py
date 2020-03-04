

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
        self.uniq_ind = np.unique(self.train_ind, axis=0)

        self.num_events = len(self.train_ind)
        self.num_entries = len(self.uniq_ind)

        #self.log_file_name = init_config['log_file_name']
        self.init_U = init_config['U']

        self.batch_size_event = init_config['batch_size_event']
        self.batch_size_entry = init_config['batch_size_entry']

        self.learning_rate = init_config['learning_rate']

        # VI Sparse GP
        self.B = init_config['inducing_B']  # init with k-means, [len_B, rank]
        self.len_B = len(self.B)

        self.GP_SCOPE_NAME = "gp_params"
        #GP parameters
        with tf.variable_scope( self.GP_SCOPE_NAME):
            # Embedding params
            self.tf_U = [tf.Variable(self.init_U[k], dtype=FLOAT_TYPE) for k in range(self.nmod)]

            # pseudo inputs
            self.tf_B = tf.Variable(self.B, dtype=FLOAT_TYPE)

            # pseudo outputs
            self.tf_mu_alpha = tf.Variable(np.zeros([self.len_B, 1]), dtype=FLOAT_TYPE)
            self.tf_Ltril_alpha = tf.Variable(np.eye(self.len_B), dtype=FLOAT_TYPE)
            self.tf_log_lengthscale_alpha = tf.Variable(np.zeros([self.B.shape[1], 1]), dtype=FLOAT_TYPE)
            self.tf_log_amp_alpha = tf.Variable( 0, dtype=FLOAT_TYPE)


        self.Kmm_alpha = utils_funcs.kernel_cross_tf( self.tf_B, self.tf_B, self.tf_log_amp_alpha, self.tf_log_lengthscale_alpha)
        self.Kmm_alpha = self.Kmm_alpha + MATRIX_JITTER * tf.linalg.eye( self.len_B,dtype=FLOAT_TYPE)
        self.Var_alpha = self.tf_Ltril_alpha @ tf.transpose( self.tf_Ltril_alpha)

        # KL terms
        self.KL_alpha = utils_funcs.KL_q_p_tf(self.Kmm_alpha,self.tf_Ltril_alpha, self.tf_mu_alpha,
                                              self.len_B)

        # Integral Term
        # sum_i < int_0^T lam_i>
        # placeholders
        self.batch_entry_ind = tf.placeholder(dtype=tf.int32, shape=[self.batch_size_entry, self.nmod])
        self.batch_event_ind = tf.placeholder(dtype=tf.int32, shape=[self.batch_size_event, self.nmod])
        self.batch_event_y = tf.placeholder(dtype=FLOAT_TYPE, shape=[self.batch_size_event, 1])

        self.X_entries = utils_funcs.concat_embeddings(self.tf_U, self.batch_entry_ind)
        self.X_events = utils_funcs.concat_embeddings(self.tf_U, self.batch_event_ind)

        self.tf_T = tf.constant(self.train_y[-1][0] - self.train_y[0][0], dtype=FLOAT_TYPE)
        self.tf_T0 = tf.constant( self.train_y[0][0], dtype=FLOAT_TYPE)
        self.tf_T1 = tf.constant( self.train_y[-1][0], dtype=FLOAT_TYPE)

        # sample posterior base rate ( f )
        self.Knm_entries = utils_funcs.kernel_cross_tf( self.X_entries, self.tf_B, self.tf_log_amp_alpha, self.tf_log_lengthscale_alpha)
        self.gp_base_rate_entries = utils_funcs.sample_pst_f_tf( self.tf_mu_alpha, self.tf_Ltril_alpha, self.Kmm_alpha, self.Knm_entries,
                                                                 self.tf_log_amp_alpha, MATRIX_JITTER)
        self.base_rate_entries = tf.exp( self.gp_base_rate_entries)

        #int term 1, using entryEvent
        self.int_part1 = self.num_entries / self.batch_size_entry * self.tf_T * tf.reduce_sum( self.base_rate_entries)


        # sample event base rate
        self.Knm_events = utils_funcs.kernel_cross_tf(self.X_events, self.tf_B, self.tf_log_amp_alpha,
                                                          self.tf_log_lengthscale_alpha)
        self.gp_base_rate_events = utils_funcs.sample_pst_f_tf(self.tf_mu_alpha, self.tf_Ltril_alpha, self.Kmm_alpha,
                                                                   self.Knm_events,  self.tf_log_amp_alpha,MATRIX_JITTER)

        # event sum term 1
        self.event_sum_part1 =  self.num_events / self.batch_size_event *tf.reduce_sum( self.gp_base_rate_events)

        self.ELBO = self.event_sum_part1 - self.int_part1 - self.KL_alpha
        self.neg_ELBO = - self.ELBO
        self.ELBO_hist = []

        # setting
        self.min_opt = tf.train.AdamOptimizer(self.learning_rate)
        #self.min_opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        self.min_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.GP_SCOPE_NAME)
        #print( self.min_params) ##
        self.min_step = self.min_opt.minimize(self.neg_ELBO, var_list=self.min_params)

        # GPU settings
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.entries_ind_gnrt = utils_funcs.DataGenerator(self.uniq_ind)
        self.event_ind_y_gnrt = utils_funcs.DataGenerator(self.train_ind, self.train_y)

        self.isTestGraphInitialized = False


    def train(self, steps = 1, print_every = 100, test_every = False,
              val_error = False, val_all_ind = None, val_all_y = None,
              val_test_ind = None, val_test_y = None, verbose = False):
        print('start')
        for step in range( 1, steps + 1):
            if step % print_every == 0:
                print( "step = %d " %  step )

            # max step=============>>
            batch_entries_ind = self.entries_ind_gnrt.draw_next( self.batch_size_entry)
            batch_event_ind, batch_event_y = self.event_ind_y_gnrt.draw_next( self.batch_size_event)

            train_feed_dict = { self.batch_entry_ind : batch_entries_ind,
                                self.batch_event_ind : batch_event_ind, self.batch_event_y : batch_event_y}


            _, ELBO_ret_min_step, int_part1, sum_part1, gp_base_rate_event = self.sess.run(
                [self.min_step, self.neg_ELBO, self.int_part1, self.event_sum_part1, self.gp_base_rate_events], feed_dict=train_feed_dict)
            self.ELBO_hist.append( ELBO_ret_min_step)


            if step % print_every == 0:
                print("neg ELBO min step = %g, int_part1 = %g, sum_part1 = %g, max gp = %g, min gp = %g"
                      % ( ELBO_ret_min_step, int_part1, sum_part1, np.max( gp_base_rate_event), np.min( gp_base_rate_event) ))
            #<<===================End max step


            if step % print_every == 0:
                amp_alpha = self.check_vars( [ self.tf_log_amp_alpha])
                amp_alpha = np.exp( amp_alpha)
                print('amp_alpha = %g' % ( amp_alpha))

        return self

    def create_standAlone_test_graph(self, test_ind, test_y):
        print("Create testing graph")
        self.test_ind = test_ind
        self.test_y = test_y

        self.ind_uniq_test = np.unique(test_ind, axis=0)
        self.num_uniq_ind_test = len( self.ind_uniq_test)
        self.num_test_events = len(test_ind)

        # Integral Term
        # sum_i < int_0^T lam_i>
        # placeholders
        self.entry_ind_test = tf.constant( self.ind_uniq_test, dtype=tf.int32 )
        self.event_ind_test = tf.constant( self.test_ind, dtype=tf.int32)
        self.event_y_test = tf.constant( self.test_y, dtype=FLOAT_TYPE)

        self.X_entries_test = utils_funcs.concat_embeddings(self.tf_U, self.entry_ind_test)
        self.X_events_test = utils_funcs.concat_embeddings(self.tf_U, self.event_ind_test)

        self.tf_T_test = tf.constant(self.test_y[-1][0] - self.test_y[0][0], dtype=FLOAT_TYPE)

        # sample posterior base rate ( f )
        self.Knm_entries_test = utils_funcs.kernel_cross_tf(self.X_entries_test, self.tf_B, self.tf_log_amp_alpha,
                                                       self.tf_log_lengthscale_alpha)
        self.gp_base_rate_entries_test = utils_funcs.sample_pst_f_tf_MLE(self.tf_mu_alpha, self.Kmm_alpha,
                                                                self.Knm_entries_test)
        self.base_rate_entries_test = tf.exp(self.gp_base_rate_entries_test)

        # int term 1, using entryEvent
        self.int_part_test = self.tf_T_test * tf.reduce_sum(self.base_rate_entries_test)

        # sample event base rate
        self.Knm_events_test = utils_funcs.kernel_cross_tf(self.X_events_test, self.tf_B, self.tf_log_amp_alpha,
                                                      self.tf_log_lengthscale_alpha)
        self.gp_base_rate_events_test = utils_funcs.sample_pst_f_tf_MLE(self.tf_mu_alpha, self.Kmm_alpha,self.Knm_events_test)

        # event sum term 1
        self.event_sum_test = tf.reduce_sum(self.gp_base_rate_events_test)
        self.llk_test = self.event_sum_test - self.int_part_test

        self.isTestGraphInitialized = True

        return self

    def test(self, verbose = False):
        if not self.isTestGraphInitialized:
            raise NameError("Test Graph hasn't been initialized")


        test_feed_dict = { self.entry_ind_test:self.ind_uniq_test, self.event_ind_test:self.test_ind}
        int_term, eventsum_term, test_llk = self.sess.run( [ self.int_part_test, self.event_sum_test, self.llk_test], feed_dict= test_feed_dict)

        return test_llk, int_term , eventsum_term


    def check_vars(self, var_list):
        batch_entries_ind = self.entries_ind_gnrt.draw_last()
        batch_event_ind, batch_event_y = self.event_ind_y_gnrt.draw_last()

        train_feed_dict = {self.batch_entry_ind: batch_entries_ind,
                           self.batch_event_ind: batch_event_ind, self.batch_event_y: batch_event_y}

        ret = self.sess.run( var_list, feed_dict=train_feed_dict)
        return ret


def test_data_set():
    (ind, y), (train_ind, train_y), (test_ind, test_y) = utils_funcs.load_dataSet(arg_data_name, '../Data')

    nmod = ind.shape[1]
    nvec = np.max(ind, axis=0) + 1

    T = y[-1][0] - y[0][0]
    print('T = %g' % T)

    R = arg_rank
    U = [np.random.rand(nvec[k], R)  for k in range(nmod)]

    init_config = {}
    init_config['U'] = U
    init_config['batch_size_event'] = 128
    init_config['batch_size_entry'] = 128

    init_config['learning_rate'] = arg_lr

    len_B = 128  # event

    print('lauching Kmeans')
    B = utils_funcs.init_base_gp_pseudo_inputs(U, train_ind, len_B)
    print('Kmeans end')

    # VI Sparse GP
    init_config['inducing_B'] = B  # init with k-means, [len_B, rank]

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

    utils_funcs.log_results('GP_PTF.txt', arg_data_name, arg_rank, arg_lr, test_llk)
    model.sess.close()

if __name__ == '__main__':
    test_data_set()











