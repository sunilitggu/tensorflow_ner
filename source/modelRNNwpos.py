import tensorflow as tf
import numpy as np

class RNN_CRF_NER(object):

	def __init__(self, 
			len_worddict, 
			len_posdict, 
			len_tagdict, 
			len_chardict, 
			charMax, 
			sentMax, 
			w_emb_size, 
			c_emb_size, 
			wv, 
			num_filters=100, 
			cwes=10, 
			pes=10, 
			l2_reg_lambda=0.0, 
			charmodel='lstm'):


		tf.reset_default_graph()

		self.sent_len = tf.placeholder(tf.int64, [None], name='sent_len')
		self.word_len = tf.placeholder(tf.int64, [None,None], name='word_len')
		self.word  = tf.placeholder(tf.int32, [None, None], name="word")		# NxM
		self.char = tf.placeholder(tf.int32,[None, None, None], name="char")		# NxMxC
		self.pos  = tf.placeholder(tf.int32, [None, None], name="pos")			# NxM
		self.input_y = tf.placeholder(tf.int32, [None, None], name="input_y")		# NxM
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		#Embeddings Diclaration
#		W_wemb = tf.Variable(tf.random_uniform([len_worddict, w_emb_size], -1.0, +1.0))
		W_wemb = tf.Variable(wv)
		W_chemb = tf.Variable(tf.random_uniform([len_chardict, c_emb_size], -1.0, +1.0))
		W_pemb = tf.Variable(tf.random_uniform([len_posdict, pes], -1.0, +1.0))
 
		#charecter level word embedding
		X_char = tf.nn.embedding_lookup(W_chemb, self.char)			# NxMxCx50
		print 'X_char', X_char.get_shape()

		X_char = tf.reshape(X_char, [-1, charMax, c_emb_size])			# NMxCx50

		charlen= tf.reshape(self.word_len, [-1])				# NM

		if (charmodel == 'lstm'):
    		    print 'Character LSTM initiated'
		    with tf.variable_scope('char_lstm'):	
			cell_f1 = tf.contrib.rnn.LSTMCell(num_units=cwes, state_is_tuple=True)
			cell_b1 = tf.contrib.rnn.LSTMCell(num_units=cwes, state_is_tuple=True)
			outputs1, states1 = tf.nn.bidirectional_dynamic_rnn(
									cell_fw	=cell_f1, 
									cell_bw	=cell_b1, 
									dtype	=tf.float32, 
									sequence_length= charlen, 
									inputs	=X_char
								)

			output_fw1, output_bw1 = outputs1				# NMxCX10
			states_fw1, states_bw1 = states1				# NMx10

			states_fw = tf.reshape(states_fw1.h, [-1, sentMax, cwes])	# NxMx10
			states_bw = tf.reshape(states_bw1.h, [-1, sentMax, cwes])	# NxMx10
			self.emb_c = tf.concat([states_fw, states_bw], 2 )			# NxMx20
			print 'emb_c', self.emb_c.get_shape()

		elif charmodel == 'cnn': 
		    print 'Character CNN initiated'
		    with tf.variable_scope('char_cnn'):
			filter_sizes=[2,3]			
			pooled_outputs = []
			X_char = tf.expand_dims(X_char, -1)
			for i, filter_size in enumerate(filter_sizes):
				filter_shape = [filter_size, c_emb_size, 1, cwes]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_c")
				b = tf.Variable(tf.constant(0.1, shape=[cwes]), name="b_c")
				conv = tf.nn.conv2d(X_char, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        			# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 	# shape (MN, 19, 1, 70)
				# print "h ", h.get_shape

				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(h, 
						ksize=[1, charMax - filter_size + 1, 1, 1], 
						strides=[1, 1, 1, 1], 
						padding='VALID', 
						name="pool")
				pooled_outputs.append(pooled)	
			num_filters_total = cwes * len(filter_sizes)
			h_pool = tf.concat(pooled_outputs, 3)					#shape= (MN, 1, 1, 210)			 
			self.emb_c = tf.reshape(h_pool, [-1, sentMax, num_filters_total])		#shape =(M, N, 210)


		#lookup layer
		emb0 = tf.nn.embedding_lookup(W_wemb, self.word)			# word embedding 		NxMx100
		emb2 = tf.nn.embedding_lookup(W_pemb, self.pos)

		if charmodel == 'cnn' or charmodel == 'lstm':
			X = tf.concat([emb0, self.emb_c, emb2], 2)				# (N, M, 130)
		else:
			#X = tf.concat([emb0, emb2], 2)					# (N, M, 110)
			X = emb0

		print 'X ', X.get_shape()

		#LSTM layer
		with tf.variable_scope('word_lstm'):
			cell_f1 = tf.contrib.rnn.LSTMCell(num_units=num_filters, state_is_tuple=True)
			cell_b1 = tf.contrib.rnn.LSTMCell(num_units=num_filters, state_is_tuple=True)
			outputs1, states1 = tf.nn.bidirectional_dynamic_rnn(
									cell_fw	=cell_f1, 
									cell_bw	=cell_b1, 
									dtype	=tf.float32, 	
									sequence_length=self.sent_len, 
									inputs	=X
								)

			output_fw1, output_bw1 = outputs1				# NxMx100
			states_fw1, states_bw1 = states1
			h_rnn = tf.concat([output_fw1, output_bw1], 2)			# NxMx200
			h_rnn = tf.reshape(h_rnn, [-1, 2*num_filters])			# NMx200	

		h_rnn = tf.nn.dropout(h_rnn, self.dropout_keep_prob)

		#Fully connected layer
		h_rnn = tf.tanh(h_rnn)
		W_1 = tf.get_variable("W_1", [2*num_filters, len_tagdict])		# 200x3
		b_1 = tf.Variable(tf.constant(0.1, shape=[len_tagdict]), name="b_1")	# 3

		scores = tf.nn.xw_plus_b(h_rnn, W_1, b_1, name="scores")		# NMx3
		self.unary_scores = tf.reshape(scores, [-1, sentMax, len_tagdict]) 	# NxMx3
    		self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
										self.unary_scores, 
										self.input_y, 
										self.sent_len
										)
		
		# Add a training op to tune the parameters.
    		self.loss = tf.reduce_mean(-self.log_likelihood) + l2_reg_lambda*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(b_1))
#		self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

		self.optimizer = tf.train.AdamOptimizer(1e-2)
		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
		session_conf = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)  
 		self.sess.run(tf.global_variables_initializer())
		self.step = 0


	def train_step(self, t_w, t_p, t_t, t_s, t_ch, t_ch_len, drop_out=1.0):
		self.step += 1
	
		feed_dict = {
				self.word 	:t_w,
				self.pos	:t_p,
				self.sent_len 	:t_s,
				self.char	:t_ch,
				self.word_len	:t_ch_len,
				self.dropout_keep_prob: drop_out,
				self.input_y 	:t_t
	    			}
   		_, loss = self.sess.run([self.train_op, self.loss], feed_dict)

    		print ("step "+str(self.step) + " loss "+str(loss) )
		return loss


	def test_step(self, t_w, t_p, t_t, t_s, t_ch, t_ch_len, drop_out=1.0):
		
		feed_dict = {
				self.word 	:t_w,
				self.char	:t_ch,
				self.pos	:t_p,
				self.sent_len 	:t_s,
				self.word_len	:t_ch_len,	
				self.dropout_keep_prob: drop_out,
				self.input_y 	:t_t
	    			}
   		loss, u_s, t_param = self.sess.run([self.loss, self.unary_scores, self.transition_params], feed_dict)

		pred_label = []
		correct_labels = 0
		total_labels = 0
		for u_s_, t_t_, t_s_ in zip(u_s, t_t, t_s):		
			u_s_ = u_s_[:t_s_]  								# Remove padding from the scores sequence.
			t_t_ = t_t_[:t_s_]								# Remove padding from the tag sequence.			
			viterbi_sequence,_ = tf.contrib.crf.viterbi_decode(u_s_, t_param)		# highest scoring sequence.
			pred_label.append(viterbi_sequence)
			# Evaluate word-level accuracy.
			correct_labels += np.sum( np.equal(viterbi_sequence, t_t_ ) )
          		total_labels += t_s_

        	accuracy = 100.0 * correct_labels / float(total_labels)
		print "Accuracy in test set", accuracy
		return accuracy, pred_label 


	 
	def getCharWordEmb(self, char_list, char_len):
		# char_list = [] []
		# char_len = [] [] []
		feed_dict = {
				self.char	:char_list,
				self.word_len	:char_len,	
	    			}
   		emb = self.sess.run([self.emb_c], feed_dict)
		return emb
