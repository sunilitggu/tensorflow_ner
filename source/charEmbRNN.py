import tensorflow as tf
import numpy as np

class RNN_CRF_NER(object):

	def __init__(self, len_worddict, len_chardict, charMax, c_emb_size, w_emb_size, cwes=10 ):

		tf.reset_default_graph()

		self.word_len = tf.placeholder(tf.int64, [None], name='word_len')
		self.char = tf.placeholder(tf.int32,[None, None], name="char")		# MxC

		#Embeddings Diclaration
		W_wemb = tf.Variable(tf.random_uniform([len_worddict, w_emb_size], -1.0, +1.0))

		W_chemb = tf.Variable(tf.random_uniform([len_chardict, c_emb_size], -1.0, +1.0))

		X_char = tf.nn.embedding_lookup(W_chemb, self.char)			# MxCx50

		print 'X_char', X_char.get_shape()
		with tf.variable_scope('char_lstm'):	
	    		cell_f1 = tf.contrib.rnn.LSTMCell(num_units=cwes, state_is_tuple=True)
			cell_b1 = tf.contrib.rnn.LSTMCell(num_units=cwes, state_is_tuple=True)
			outputs1, states1 = tf.nn.bidirectional_dynamic_rnn(
								cell_fw	=cell_f1, 
								cell_bw	=cell_b1, 
								dtype	=tf.float32, 
								sequence_length= self.word_len, 
								inputs	=X_char
							)

			output_fw1, output_bw1 = outputs1				# MxCX10
			states_fw1, states_bw1 = states1				# Mx10

		states_fw = states_fw1.h					# Mx10
		states_bw = states_bw1.h					# Mx10

		self.emb_c = tf.concat([states_fw, states_bw], 1)		# Mx20
		self.emb_c = tf.reshape(self.emb_c, [-1, cwes*2])

		print 'emb_c', self.emb_c.get_shape()
	 
	 
	def getCharWordEmb(self, char_list, char_len):

		feed_dict = {
				self.char	:char_list,
				self.word_len	:char_len,	
	    			}

   		emb = self.sess.run(self.emb_c, feed_dict)

		return emb



