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
		h_pool = tf.concat(pooled_outputs, 3)						#shape= (MN, 1, 1, 210)			 
		self.emb_c = tf.reshape(h_pool, [-1, cwes*2])

		print 'emb_c', self.emb_c.get_shape()
	 
	def getCharWordEmb(self, char_list, char_len):

		feed_dict = {
				self.char	:char_list,
				self.word_len	:char_len,	
	    			}

   		emb = self.sess.run(self.emb_c, feed_dict)

		return emb



