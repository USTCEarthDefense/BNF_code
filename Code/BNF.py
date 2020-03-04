import utils_funcs
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import joblib as jb
from utils_funcs import FLOAT_TYPE, MATRIX_JITTER, DELTA_JITTER
import sys
import os
from tensorflow import keras

#np.random.seed(47)
#tf.set_random_seed(47)

# run as
print("usage : python *.py gpu=0 rank=5 dataset=article lr=0.001")

print('start')
print(sys.argv)
# parse args
py_name = sys.argv[0]
args = sys.argv[1:]
args_dict = {}
for arg_pair in args:
    arg, val_str = arg_pair.split('=')
    val_str = val_str.strip()
    args_dict[arg] = val_str

arg_gpu_idx_str = args_dict['gpu']
arg_rank = int(args_dict['rank'])
arg_data_name = args_dict['dataset']
arg_lr = float(args_dict['lr'])

print('gpu index = %s' % arg_gpu_idx_str)
print('rank = %d' % arg_rank)
print('learning rate = %e' % arg_lr)


class NTF_HP:
    def __init__(self, train_ind, train_y, init_config):
        self.train_ind = train_ind
        self.train_y = train_y
        self.nmod = train_ind.shape[1]
        self.uniq_ind = np.unique(self.train_ind, axis=0)

        self.num_events = len(self.train_ind)
        self.num_entries = len(self.uniq_ind)

        # self.log_file_name = init_config['log_file_name']
        self.init_U = init_config['U']
        self.rank = self.init_U[0].shape[1]

        self.batch_size_event = init_config['batch_size_event']
        self.batch_size_inner_event = init_config['batch_size_inner_event']
        self.batch_size_entry = init_config['batch_size_entry']
        self.batch_size_entryEvent = init_config['batch_size_entryEvent']

        self.learning_rate = init_config['learning_rate']

        # VI Sparse GP
        self.B = init_config['inducing_B']  # init with k-means, [len_B, rank]
        self.len_B = len(self.B)

        self.GP_SCOPE_NAME = "gp_params"
        # GP parameters
        with tf.variable_scope(self.GP_SCOPE_NAME):
            # Embedding params
            self.tf_U = [tf.Variable(self.init_U[k], dtype=FLOAT_TYPE,) for k in range(self.nmod)]
            #keras.initializers.
            # pseudo inputs
            self.tf_B = tf.Variable(self.B, dtype=FLOAT_TYPE)

            # pseudo outputs
            self.tf_mu_alpha = tf.Variable(np.random.randn(self.len_B, 1) * 0.1, dtype=FLOAT_TYPE)
            #self.tf_Ltril_alpha = tf.Variable(np.eye(self.len_B), dtype=FLOAT_TYPE)
            self.tf_Ltril_alpha =  tf.linalg.band_part(  tf.Variable(np.eye(self.len_B) * 0.5, dtype=FLOAT_TYPE), -1, 0)

            self.tf_log_lengthscale_alpha = tf.Variable(np.zeros([self.B.shape[1], 1]), dtype=FLOAT_TYPE)
            self.tf_log_amp_alpha = tf.Variable( init_config[ 'log_amp_alpha'], dtype=FLOAT_TYPE)

            self.tf_log_lengthscale_delta = tf.Variable(np.zeros([self.B.shape[1], 1]), dtype=FLOAT_TYPE)

            self.tf_log_amp_delta = tf.Variable(init_config['log_amp_delta'], dtype=FLOAT_TYPE)
            #self.tf_log_amp_delta = self.tf_log_amp_alpha

            self.tf_log_lengthscale_trig = tf.Variable(np.zeros([self.B.shape[1], 1]), dtype=FLOAT_TYPE)
            self.tf_log_amp_trig = tf.Variable(init_config['log_amp_trig'], dtype=FLOAT_TYPE)
            #self.tf_log_amp_trig = tf.constant( -50, dtype=FLOAT_TYPE)

        self.Kmm_alpha = utils_funcs.kernel_cross_tf(self.tf_B, self.tf_B, self.tf_log_amp_alpha,
                                                     self.tf_log_lengthscale_alpha)
        self.Kmm_alpha = self.Kmm_alpha + MATRIX_JITTER * tf.linalg.eye(self.len_B, dtype=FLOAT_TYPE)
        self.Var_alpha = self.tf_Ltril_alpha @ tf.transpose(self.tf_Ltril_alpha)

        # KL terms
        self.KL_alpha = utils_funcs.KL_q_p_tf(self.Kmm_alpha, self.tf_Ltril_alpha, self.tf_mu_alpha, self.len_B)

        # Integral Term
        # sum_i < int_0^T lam_i>
        # placeholders
        self.batch_entry_ind = tf.placeholder(dtype=tf.int32, shape=[self.batch_size_entry, self.nmod])
        self.batch_entryEvent_ind = tf.placeholder(dtype=tf.int32, shape=[self.batch_size_entryEvent, self.nmod])
        self.batch_entryEvent_y = tf.placeholder(dtype=FLOAT_TYPE, shape=[self.batch_size_entryEvent, 1])

        self.batch_event_ind = tf.placeholder(dtype=tf.int32, shape=[self.batch_size_event, self.nmod])
        self.batch_event_y = tf.placeholder(dtype=FLOAT_TYPE, shape=[self.batch_size_event, 1])

        self.batch_inner_event_ind = tf.placeholder(dtype=tf.int32, shape=[self.batch_size_inner_event, self.nmod])
        self.batch_inner_event_y = tf.placeholder(dtype=FLOAT_TYPE, shape=[self.batch_size_inner_event, 1])

        self.X_entries = utils_funcs.concat_embeddings(self.tf_U, self.batch_entry_ind)
        self.X_entryEvents = utils_funcs.concat_embeddings(self.tf_U, self.batch_entryEvent_ind)
        self.X_events = utils_funcs.concat_embeddings(self.tf_U, self.batch_event_ind)
        self.X_inner_events = utils_funcs.concat_embeddings(self.tf_U, self.batch_inner_event_ind)

        self.tf_T = tf.constant(self.train_y[-1][0] - self.train_y[0][0], dtype=FLOAT_TYPE)
        self.tf_T0 = tf.constant(self.train_y[0][0], dtype=FLOAT_TYPE)
        self.tf_T1 = tf.constant(self.train_y[-1][0], dtype=FLOAT_TYPE)

        # sample posterior base rate ( f )
        self.Knm_entries = utils_funcs.kernel_cross_tf(self.X_entries, self.tf_B, self.tf_log_amp_alpha,
                                                       self.tf_log_lengthscale_alpha)
        self.gp_base_rate_entries = utils_funcs.sample_pst_f_tf(self.tf_mu_alpha, self.tf_Ltril_alpha, self.Kmm_alpha,
                                                                self.Knm_entries,
                                                                self.tf_log_amp_alpha, MATRIX_JITTER)
        self.base_rate_entries = tf.exp(self.gp_base_rate_entries)

        # int term 1, using entryEvent
        self.int_part1 = self.num_entries / self.batch_size_entry * self.tf_T * tf.reduce_sum(self.base_rate_entries)

        # term 2
        # Sample posterior decay rate (delta)
        self.delta_entries = 1.0 / (
                utils_funcs.kernel_cross_tf(self.X_entries, self.X_entryEvents, self.tf_log_amp_delta,
                                            self.tf_log_lengthscale_delta) + DELTA_JITTER)
        self.T_sub_y = self.tf_T1 - self.batch_entryEvent_y  # [N_e, 1]
        self.lag_effect = (1.0 - tf.exp(
            -self.delta_entries * tf.transpose(self.T_sub_y))) / self.delta_entries  # [ N_i, N_e]
        self.K_trig = utils_funcs.kernel_cross_tf(self.X_entries, self.X_entryEvents, self.tf_log_amp_trig,
                                                  self.tf_log_lengthscale_trig)  # [S,Q]
        self.Trig_Mat = self.K_trig * self.lag_effect
        # integral term 2
        self.int_part2 = self.num_entries / self.batch_size_entry * self.num_events / self.batch_size_entryEvent * tf.reduce_sum(
            self.Trig_Mat)

        # sample event base rate
        self.Knm_events = utils_funcs.kernel_cross_tf(self.X_events, self.tf_B, self.tf_log_amp_alpha,
                                                      self.tf_log_lengthscale_alpha)
        self.gp_base_rate_events = utils_funcs.sample_pst_f_tf(self.tf_mu_alpha, self.tf_Ltril_alpha, self.Kmm_alpha,
                                                               self.Knm_events, self.tf_log_amp_alpha, MATRIX_JITTER)
        self.base_rate_events = tf.exp(self.gp_base_rate_events)

        self.event_delay = self.batch_event_y - tf.transpose(self.batch_inner_event_y)
        self.valid_event_delay = tf.cast(self.event_delay > 0, FLOAT_TYPE)
        self.event_delay = self.event_delay * self.valid_event_delay
        self.delta_eventSum = 1.0 / (
                utils_funcs.kernel_cross_tf(self.X_events, self.X_inner_events, self.tf_log_amp_delta,
                                            self.tf_log_lengthscale_delta) + DELTA_JITTER)
        self.event_delay_effect = tf.exp(- self.delta_eventSum * self.event_delay)
        self.K_trig_event = utils_funcs.kernel_cross_tf(self.X_events, self.X_inner_events, self.tf_log_amp_trig,
                                                        self.tf_log_lengthscale_trig)
        self.trig_mat_event = self.K_trig_event * self.event_delay_effect * self.valid_event_delay
        self.trig_effects_eventSum = tf.reduce_sum( self.trig_mat_event, axis=1, keepdims=True)

        # Bias est of event rates
        self.event_rates = self.base_rate_events + self.num_events / self.batch_size_inner_event * self.trig_effects_eventSum

        # sum term
        self.event_sum_term = self.num_events / self.batch_size_event * tf.reduce_sum(tf.log(self.event_rates))

        sqr_U = [tf.reduce_sum(U_i * U_i) for U_i in self.tf_U]
        self.U_kernel_mag = tf.reduce_sum(sqr_U)

        ###
        self.train_ELBO = 1.0 * self.event_sum_term - self.int_part1 - self.int_part2 - self.KL_alpha  # - self.KL_delta
        ###

        self.train_ELBO_hist = []

        # setting
        self.min_opt = tf.train.AdamOptimizer(self.learning_rate)
        self.min_step = self.min_opt.minimize(- self.train_ELBO)

        # GPU settings
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        self.sess.run(tf.global_variables_initializer())

        self.entries_ind_gnrt = utils_funcs.DataGenerator(self.uniq_ind)
        self.event_ind_y_gnrt = utils_funcs.DataGenerator(self.train_ind, self.train_y)
        self.entryEvent_ind_y_gnrt = utils_funcs.DataGenerator(self.train_ind, self.train_y)
        self.inner_event_ind_y_gnrt = utils_funcs.DataGenerator(self.train_ind, self.train_y)

        self.isTestGraphInitialized = False

    def train(self, steps=1, print_every=100, test_every=False,
              val_error=False, val_all_ind=None, val_all_y=None,
              val_test_ind=None, val_test_y=None, verbose=False):
        print('start')
        for step in range(1, steps + 1):
            if step % print_every == 0:
                print("step = %d " % step)

            batch_entries_ind = self.entries_ind_gnrt.draw_next(self.batch_size_entry)
            batch_event_ind, batch_event_y = self.event_ind_y_gnrt.draw_next(self.batch_size_event)
            batch_entryEvent_ind, batch_entryEvent_y = self.entryEvent_ind_y_gnrt.draw_next(self.batch_size_entryEvent)
            batch_inner_event_ind, batch_inner_event_y = self.inner_event_ind_y_gnrt.draw_next(
                self.batch_size_inner_event)

            train_feed_dict = {self.batch_entry_ind: batch_entries_ind,
                               self.batch_event_ind: batch_event_ind, self.batch_event_y: batch_event_y,

                               self.batch_entryEvent_ind: batch_entryEvent_ind,
                               self.batch_entryEvent_y: batch_entryEvent_y,
                               self.batch_inner_event_ind: batch_inner_event_ind,
                               self.batch_inner_event_y: batch_inner_event_y}

            _, ELBO_ret_min_step, KL_alpha, int_part1, int_part2, event_sum_term, entry_base_rate, delta_entries, = self.sess.run(
                [self.min_step, self.train_ELBO, self.KL_alpha, self.int_part1, self.int_part2, self.event_sum_term,
                 self.base_rate_entries, self.delta_entries],
                feed_dict=train_feed_dict, options=self.run_options
            )

            self.train_ELBO_hist.append(ELBO_ret_min_step)

            if step % print_every == 0:
                int_term = int_part1 + int_part2
                int_event_sum_ratio = np.abs( int_term / event_sum_term)
                print(
                    "ELBO = %g, KL_alpha = %g, int_part1 = %g, int_part2 = %g, event sum =  %g, ratio = %g,  base rate max = %g, min = %g, delta max = %g, min = %g"
                    % (ELBO_ret_min_step, KL_alpha, int_part1, int_part2, event_sum_term,int_event_sum_ratio,  np.max(entry_base_rate),
                       np.min(entry_base_rate), np.max(delta_entries), np.min(delta_entries)))
            # End min step<<=======================

            if step % print_every == 0:
                amp_alpha, amp_trig, amp_delta, U_reg, = self.check_vars(
                    [self.tf_log_amp_alpha, self.tf_log_amp_trig, self.tf_log_amp_delta, self.U_kernel_mag])
                amp_alpha, amp_trig, amp_delta = np.exp([amp_alpha, amp_trig, amp_delta])
                print('amp_alpha = %g, amp_trig = %g, amp_delta = %g, U_mag = %g' % (
                amp_alpha, amp_trig, amp_delta, U_reg))

        return self

    def create_standAlone_test_graph(self, test_ind, test_y):
        print("Create testing graph")
        self.test_ind = test_ind
        self.test_y = test_y

        self.ind_uniq_test = np.unique(test_ind, axis=0)
        self.num_uniq_ind_test = len(self.ind_uniq_test)
        self.num_test_events = len(test_ind)

        self.tf_test_event_ind_test = tf.constant(self.test_ind, dtype=tf.int32)  # ALL test events
        self.tf_test_event_y_test = tf.constant(self.test_y, dtype=FLOAT_TYPE)  # ALL test events
        self.tf_batch_entries_ind_test = tf.placeholder(dtype=tf.int32, shape=[None, self.nmod])

        # integral term
        # Use full testing event term when calculating batch integral terms
        self.T_test = tf.constant(self.test_y[-1][0] - self.test_y[0][0], dtype=FLOAT_TYPE)
        self.T0_test = tf.constant(self.test_y[0][0], dtype=FLOAT_TYPE)
        self.T1_test = tf.constant(self.test_y[-1][0], dtype=FLOAT_TYPE)

        self.X_batch_entries_test = utils_funcs.concat_embeddings(self.tf_U, self.tf_batch_entries_ind_test)
        self.X_all_test_events_test = utils_funcs.concat_embeddings(self.tf_U, self.tf_test_event_ind_test)

        self.Knm_entries_test = utils_funcs.kernel_cross_tf(self.X_batch_entries_test, self.tf_B,
                                                            self.tf_log_amp_alpha, self.tf_log_lengthscale_alpha)
        self.gp_base_rate_entries_test = utils_funcs.sample_pst_f_tf_MLE(self.tf_mu_alpha, self.Kmm_alpha,
                                                                         self.Knm_entries_test)
        self.base_rate_entries_test = tf.exp(self.gp_base_rate_entries_test)

        # term1
        self.int_term1_test = tf.reduce_sum(self.base_rate_entries_test) * self.T_test

        delta_entries_test = 1.0 / (utils_funcs.kernel_cross_tf(self.X_batch_entries_test, self.X_all_test_events_test,
                                                                self.tf_log_amp_delta,
                                                                self.tf_log_lengthscale_delta) + DELTA_JITTER)
        T_sub_y_test = self.T1_test - self.tf_test_event_y_test
        lag_effect_test = (1.0 - tf.exp(- delta_entries_test * tf.transpose(T_sub_y_test))) / delta_entries_test
        self.K_trig_term3_test = utils_funcs.kernel_cross_tf(self.X_batch_entries_test, self.X_all_test_events_test,
                                                             self.tf_log_amp_trig, self.tf_log_lengthscale_trig)
        self.term3_mat = self.K_trig_term3_test * lag_effect_test

        # term3
        self.int_term3_test = tf.reduce_sum(self.term3_mat)
        self.int_term_test = self.int_term1_test + self.int_term3_test

        # event sum term
        self.tf_batch_event_ind_test = tf.placeholder(dtype=tf.int32, shape=[None, self.nmod])
        self.tf_batch_event_y_test = tf.placeholder(dtype=FLOAT_TYPE, shape=[None, 1])
        self.X_batch_event_test = utils_funcs.concat_embeddings(self.tf_U, self.tf_batch_event_ind_test)

        self.test_eventSum_Knm = utils_funcs.kernel_cross_tf(self.X_batch_event_test, self.tf_B, self.tf_log_amp_alpha,
                                                             self.tf_log_lengthscale_alpha)
        self.gp_eventSum_base_rate = utils_funcs.sample_pst_f_tf_MLE(self.tf_mu_alpha, self.Kmm_alpha,
                                                                     self.test_eventSum_Knm)
        self.eventSum_base_rate = tf.exp(self.gp_eventSum_base_rate)  # [ N_prime_batch, 1]

        self.event_delay_test = self.tf_batch_event_y_test - tf.transpose(self.tf_test_event_y_test)
        self.valid_event_delay_test = tf.cast(self.tf_batch_event_y_test > tf.transpose(self.tf_test_event_y_test),
                                              dtype=FLOAT_TYPE)
        self.event_delay_test = self.event_delay_test * self.valid_event_delay_test

        self.delta_eventSum_test = 1.0 / (
                utils_funcs.kernel_cross_tf(self.X_batch_event_test, self.X_all_test_events_test,
                                            self.tf_log_amp_delta, self.tf_log_lengthscale_delta) + DELTA_JITTER)
        self.event_delay_effect_test = tf.exp(- self.delta_eventSum_test * self.event_delay_test)

        self.K_trig_event_sum_test = utils_funcs.kernel_cross_tf(self.X_batch_event_test, self.X_all_test_events_test,
                                                                 self.tf_log_amp_trig, self.tf_log_lengthscale_trig)
        self.trig_mat_event_test = self.K_trig_event_sum_test * self.event_delay_effect_test * self.valid_event_delay_test

        self.event_rates_test = self.eventSum_base_rate + tf.reduce_sum(self.trig_mat_event_test, axis=1, keepdims=True)

        # eventsum term
        self.eventSum_term = tf.reduce_sum(tf.log(self.event_rates_test))

        self.isTestGraphInitialized = True

        return self

    def test(self, entries_batch_size, event_batch_size, verbose=False):
        if not self.isTestGraphInitialized:
            raise NameError("Test Graph hasn't been initialized")

        # Calculate entries term
        # Using full events
        if entries_batch_size > self.num_uniq_ind_test:
            entries_batch_size = self.num_uniq_ind_test

        lst_int_terms = []
        lst_term1s = []
        lst_term3s = []
        cur_idx = 0
        end_idx = cur_idx + entries_batch_size

        while cur_idx < self.num_uniq_ind_test:
            batch_entries_test = self.ind_uniq_test[cur_idx:end_idx]
            feed_dict = {self.tf_batch_entries_ind_test: batch_entries_test}

            batch_int_term, batch_int_term1, batch_int_term3 = self.sess.run(
                [self.int_term_test, self.int_term1_test, self.int_term3_test],
                feed_dict=feed_dict)

            lst_int_terms.append(batch_int_term)
            lst_term1s.append(batch_int_term1)
            lst_term3s.append(batch_int_term3)

            if verbose:
                print("int terms %d ~ %d, int_term = %g, int term1 = %g, int term3 = %g" % (
                    cur_idx, end_idx, lst_int_terms[-1], lst_term1s[-1], lst_term3s[-1]))

            cur_idx += entries_batch_size
            end_idx += entries_batch_size
            if end_idx >= self.num_uniq_ind_test:
                end_idx = self.num_uniq_ind_test

        int_term = np.sum(lst_int_terms)
        int_term1 = np.sum(lst_term1s)
        int_term3 = np.sum(lst_term3s)

        if event_batch_size > self.num_test_events:
            event_batch_size = self.num_test_events

        lst_eventSum_terms = []
        cur_idx = 0
        end_idx = cur_idx + event_batch_size

        while cur_idx < self.num_test_events:
            batch_test_event_ind = self.test_ind[cur_idx:end_idx]
            batch_test_event_y = self.test_y[cur_idx:end_idx]

            feed_dict = {self.tf_batch_event_ind_test: batch_test_event_ind,
                         self.tf_batch_event_y_test: batch_test_event_y}

            event_sum = self.sess.run(self.eventSum_term, feed_dict=feed_dict)

            lst_eventSum_terms.append(event_sum)

            if verbose:
                print("eventSum terms %d ~ %d, event_sum = %g " % (cur_idx, end_idx, lst_eventSum_terms[-1]))

            cur_idx += event_batch_size
            end_idx += event_batch_size
            if end_idx >= self.num_test_events:
                end_idx = self.num_test_events

        eventSum_term = np.sum(lst_eventSum_terms)

        test_log_P = - int_term + eventSum_term

        return test_log_P, int_term, int_term1, int_term3, eventSum_term,

    def check_vars(self, var_list):
        batch_entries_ind = self.entries_ind_gnrt.draw_last()
        batch_event_ind, batch_event_y = self.event_ind_y_gnrt.draw_last()
        batch_entryEvent_ind, batch_entryEvent_y = self.entryEvent_ind_y_gnrt.draw_last()
        batch_inner_event_ind, batch_inner_event_y = self.inner_event_ind_y_gnrt.draw_last()

        train_feed_dict = {self.batch_entry_ind: batch_entries_ind,
                           self.batch_event_ind: batch_event_ind, self.batch_event_y: batch_event_y,
                           self.batch_entryEvent_ind: batch_entryEvent_ind, self.batch_entryEvent_y: batch_entryEvent_y,
                           self.batch_inner_event_ind: batch_inner_event_ind,
                           self.batch_inner_event_y: batch_inner_event_y}

        ret = self.sess.run(var_list, feed_dict=train_feed_dict)
        return ret


def test_data_set():
    (ind, y), (train_ind, train_y), (test_ind, test_y) = utils_funcs.load_dataSet(arg_data_name, '../Data')

    nmod = ind.shape[1]
    nvec = np.max(ind, axis=0) + 1

    R = arg_rank
    U = [np.random.rand(nvec[k], R) * 1.0 for k in range(nmod)]

    init_config = {}
    init_config['U'] = U
    init_config['batch_size_event'] = 64
    init_config['batch_size_entry'] = 64

    init_config['batch_size_inner_event'] = 4096
    init_config['batch_size_entryEvent'] = 4096

    init_config['learning_rate'] = arg_lr

    init_config['log_amp_alpha'] = 0.0
    init_config['log_amp_delta'] = 0.0
    init_config['log_amp_trig'] = -3
    len_B = 128  # Base Rate


    model_config = {
        'log_amp_alpha' : init_config['log_amp_alpha'],
        'log_amp_delta' : init_config['log_amp_delta'],
        'log_amp_trig' : init_config['log_amp_trig'],
        'rank' :  arg_rank,
        'MATRIX_JITTER' : MATRIX_JITTER,
        'DELTA_JITTER': DELTA_JITTER,
        'lr' : arg_lr,
        'batch_size_event' : init_config['batch_size_event'],
        'batch_size_entry' : init_config['batch_size_entry'],
        'batch_size_inner_event' : init_config['batch_size_inner_event'],
        'batch_size_entryEvent' : init_config['batch_size_entryEvent'],
        'num_psd_points' : len_B
    }

    print('launching Kmeans')
    B = utils_funcs.init_base_gp_pseudo_inputs(U, train_ind, len_B)
    print(B.shape)
    print('Kmeans end')

    # VI Sparse GP
    init_config['inducing_B'] = B  # init with k-means, [len_B, rank]

    model = NTF_HP(train_ind, train_y, init_config)
    model.create_standAlone_test_graph(test_ind, test_y)

    steps_per_epoch = int(len(train_ind) / init_config['batch_size_event'])
    num_epoch = 50

    log_file = utils_funcs.init_log_file( './BNF.txt', arg_data_name, model_config )

    for epoch in range(1, num_epoch + 1):
        print('epoch %d\n' % epoch)
        model.train(steps_per_epoch, int(steps_per_epoch / 5))
        test_log_p, int_term, int_term1, int_term3, eventSum_term = model.test(128, 16, verbose=False)

        log_file.write( '%g\n' % test_log_p)
        log_file.flush()
        os.fsync(log_file.fileno())

    log_file.close()
    model.sess.close()

if __name__ == '__main__':
    test_data_set()
