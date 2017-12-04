import tensorflow as tf
import numpy as np
import random
import re
import collections 
import pickle
import sys
import subprocess

sys.path.append("source")
from utils import *

from modelRNN import *

embSize = 100
cembSize = 50		# char emb
cwes=10			# char level word emb
pes=10			# pos emb
charmodel='cnn'

minfreq = 2		# rear word replace c
hidden_layer_size = 250
num_epochs = 25
N = 2
drop_out = 1.0
l2_reg_lambda = 0.001
batch_size = 50

#out_file = 'tmp_results/bank_'
#results_file = 'tmp_results/results_bank.txt'

modal_name = 'saved_models/bank_'

out_file = sys.argv[1]
results_file = sys.argv[2]

#path = "dataset/DrugName/"
#train_f = "train.txt"
#test_f = "test.txt"

#path = "dataset/i2b2-2010/step2/"
#train_f = "train.txt"
#test_f = "test.txt"

#path = "dataset/DrugName/"
#train_f = "med_train.txt"
#test_f = "med_test.txt"

path = "dataset/ncbi/"
train_f = "train.txt"
test_f = "test.txt"


wefile = "/home/sunil/embeddings/glove_100d_w9_pubmed.txt"

# Reading
#train_data = DrugDataRead(path+train_f)		# return [wordlist, poslist, taglist, charlist]
#test_data = DrugDataRead(path+test_f)

train_data = DiseaseDataRead(path+train_f)		# return [wordlist, poslist, taglist, charlist]
test_data = DiseaseDataRead(path+test_f)


#Padding 
train_sent_lengths, test_sent_lengths = findSentLengths([train_data[0], test_data[0]])
sentMax = max(train_sent_lengths + test_sent_lengths)
print "max sent length", sentMax

chardata = train_data[3] + test_data[3] 
charMax =  max( [len(word) for sent in chardata for word in sent] )		#max number char in word
print 'max char length', charMax

train_sent_lengths = np.array(train_sent_lengths)
test_sent_lengths = np.array(test_sent_lengths)

train_wordlist = padList(train_data[0], sentMax)
train_poslist  = padList(train_data[1], sentMax)
train_taglist  = padList(train_data[2], sentMax, pad_symbol='O')
train_charlist, train_char_lengths = padChar(train_data[3], sentMax, charMax)	#character padding

test_wordlist = padList(test_data[0], sentMax)
test_poslist  = padList(test_data[1], sentMax)
test_taglist  = padList(test_data[2], sentMax, pad_symbol='O')
test_charlist, test_char_lengths = padChar(test_data[3], sentMax, charMax)	#character padding

#Replace rear word:
train_wordlist, test_wordlist = replaceRearWord([train_wordlist, test_wordlist], minfreq=minfreq)


#Make Dictonary 
worddict = makeDictionary(train_wordlist + test_wordlist)
posdict  = makeDictionary(train_poslist + test_poslist)
tagdict  = makeDictionary(train_taglist + test_taglist)
chardict = makeDictionaryChar(train_charlist + test_charlist)

print 'number of words in wordict', len(worddict)
print 'number of char in chardict', len(chardict)
print 'number of tag in posdict', len(posdict)
print 'tagdict', tagdict

wv,oov = readWordEmb(worddict, wefile, embSize)

# mapping
train_wordlist = mapId(train_wordlist, worddict)
train_poslist  = mapId(train_poslist, posdict)
train_taglist  = mapId(train_taglist, tagdict)
train_charlist  = mapCharId(train_charlist, chardict)

test_wordlist = mapId(test_wordlist, worddict)
test_poslist  = mapId(test_poslist, posdict)
test_taglist  = mapId(test_taglist, tagdict)
test_charlist  = mapCharId(test_charlist, chardict)

train_len = len(train_wordlist)
test_len  = len(test_wordlist)

print 'train', len(train_wordlist)

len_worddict = len(worddict)
len_posdict  = len(posdict)
len_tagdict  = len(tagdict)
len_chardict  = len(chardict)

#g = tf.Graph()
#with g.as_default():

iii = 0		

for i in [0]:
    ner = RNN_CRF_NER(len_worddict, 
			len_posdict, 
			len_tagdict, 
			len_chardict,
			charMax, 
			sentMax,           
			w_emb_size=embSize, 
			c_emb_size=cembSize, 
			wv=wv, 
			num_filters=hidden_layer_size, 
			cwes=cwes, 
			pes=pes, 
			l2_reg_lambda = l2_reg_lambda, 
			charmodel=charmodel)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    session_conf = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
    ner.sess = tf.Session(config=session_conf)  
    ner.sess.run(tf.global_variables_initializer())

    cnt=0
    for epoch in range(num_epochs):	
	shuffle_indices = np.random.permutation(np.arange(train_len))
	t_w = train_wordlist[shuffle_indices]
	t_p = train_poslist[shuffle_indices]
	t_t = train_taglist[shuffle_indices]
	t_s = train_sent_lengths[shuffle_indices]
	t_ch_len = train_char_lengths[shuffle_indices]
	t_ch = train_charlist[shuffle_indices]

	num_batches_per_epoch = int(train_len/batch_size) + 1
	for batch_num in range(num_batches_per_epoch):	
		start_index = batch_num*batch_size
		end_index = min((batch_num + 1) * batch_size, train_len)

		loss = ner.train_step( t_w[start_index:end_index], 
				t_p[start_index:end_index], 
				t_t[start_index:end_index],
				t_s[start_index:end_index],
				t_ch[start_index:end_index],
				t_ch_len[start_index:end_index]
			)

	if (epoch%N) == 0:
			
#		saver = tf.train.Saver()
#		save_path = saver.save(ner.sess, modal_name+"_%s.ckpt"%iii)		
#		iii += 1

		test_batch = 1000	
		cnt += 1
		fp = open(out_file+str(cnt), 'w')
		num_per_epoch = int(test_len/test_batch) + 1
		pred = []
		
		for bn in range(num_per_epoch):
			si = bn*test_batch
			ei = min((bn + 1) * test_batch, test_len)
			acc, p = ner.test_step( test_wordlist[si:ei],
					test_poslist[si:ei],
					test_taglist[si:ei],
					test_sent_lengths[si:ei],
					test_charlist[si:ei],
					test_char_lengths[si:ei]
				)
			pred.extend(p)

		for word_list, true_tag_list, pred_tag_list, leng in zip(test_wordlist, test_taglist, pred, test_sent_lengths):
			for w,t,p in zip(word_list[0:leng], true_tag_list[0:leng], pred_tag_list[0:leng]):
				fp.write(str(worddict[w])+' '+str(tagdict[t])+' '+str(tagdict[p])+'\n')
			fp.write('\n')

		fp.close()

fw = open(results_file, 'w')
for cnt in range(1,num_epochs/N):
	output_file_path = out_file+str(cnt)
	result = subprocess.check_output('./results/connlleval.pl < ' + output_file_path, shell=True)
	fw.write(str(result))

fw.close()

fw = open('analysis_result/clinical_oov_dict.txt','w')
for word in oov:
	fw.write(word +'\n')
fw.close()

