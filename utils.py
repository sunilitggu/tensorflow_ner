import numpy as np
import random
import csv
import re
import sys
import collections
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
import pickle

def preProcess(sent):
	sent = sent.lower()
 	sent = re.sub('\d', 'dg', sent)

	return sent


def DiseaseDataRead(fname):
	print "Input File Reading",fname
	w = open(fname, 'r') 
	fw = w.read().strip().split('\n\n')
	wordlist = []
	charlist = []
	normlist = []
	poslist = []
	chunklist = []
	genelist = []
	taglist = []
	cnt = 0
	for block in fw:
		word, norm, pos, chunk, gene, tag = block.split('\n')

		ch_word = word.split()
		char = []
		for wordc in ch_word:
			tmp = [c for c in wordc]
			if len(tmp) > 40:
				cntBigWord += 1
			 	char.append(['<','r','a','r','e','>'])
			else:
				char.append(tmp)

		word = preProcess(word)			#pre-Process
		word = word.strip().split() 
		norm = norm.strip().split()
		pos = pos.strip().split()
		chunk = chunk.strip().split()
		gene = gene.strip().split()
		tag = tag.strip().split()
		if len(word) == len(norm) and len(norm) == len(pos) and len(pos) == len(chunk) and len(chunk) == len(gene) and len(gene)==len(tag):
			wordlist.append(word)
			normlist.append(norm)
			poslist.append(pos)
			chunklist.append(chunk)
			genelist.append(gene)
			charlist.append(char)

			tagl = []
			for t in tag:
				if t.startswith('B-') :
					tagl.append('B-Dis')
				elif t.startswith('I-'):
					tagl.append('I-Dis')
				else:
					tagl.append(t)

			taglist.append(tagl)
		else:
			cnt += 1
			print 'number of words and tags are not same'
			print word
	#data = [wordlist, normlist, poslist, chunklist, genelist, taglist, charlist]
	data = [wordlist, poslist, taglist, charlist]
	print "number of sents",len(wordlist)

	return  data



def DrugDataRead(fname):
	print "Input File Reading",fname
	w = open(fname, 'r') 
	fw = w.read().strip().split('\n\n')
	wordlist = []
	charlist = []
	poslist = []
	taglist = []
	cnt = 0
	cntShort = 0
	cntBigWord = 0
	for block in fw:
		word, pos, tag = block.split('\n')
		if len( word.split())  > 100 :
			cntShort += 1
			continue

#		tag = tag.replace('0','O')		#Drug dataset was having other with 0 tag
		ch_word = word.split()
		char = []
		for wordc in ch_word:
			tmp = [c for c in wordc]
			if len(tmp) > 40:
				cntBigWord += 1
			 	char.append(['<','r','a','r','e','>'])
				
			else:
				char.append(tmp)

		word = preProcess(word)			#pre-Process
		word = word.strip().split() 
		pos = pos.strip().split()
		tag = tag.strip().split()
		if len(word) == len(pos) and len(pos) == len(tag):
			wordlist.append(word)
			poslist.append(pos)
			charlist.append(char)
			taglist.append(tag)
		else:
			cnt += 1
			print 'number of words and tags are not same'
			print word
			print pos
			print tag
			sys.exit()

	data = [wordlist, poslist, taglist, charlist]		#wordlist[Sent X Word] , charlist[Sent X Word X Char]

	
	print "number of sents",len(wordlist)
	print 'Number sents removes', cntShort
	print 'Number word removes', cntBigWord

	return  data


#2-d data
def makeDictionary(wordlist):
	worddict = []
	for sent in wordlist:		
		for word in sent:
			if word not in worddict:
				worddict.append(word)
	
	return worddict

#3-d data
def makeDictionaryChar(charlist):
	chardict = []
	for sent in charlist:
		for word in sent:
			for char in word:
				if char not in chardict:
					chardict.append(char)
	return chardict

#returns : 2d array
def mapId(wordlist, worddict):
#	print len(worddict)
	T = []
	for sent in wordlist:
		t = []
		for word in sent:
	#		print word
			t.append( worddict.index(word) )
		T.append(t)
	T = np.array(T)
	return T

#returns : 3d array
def mapCharId(charlist, chardict):
	T = []
	for sent in charlist:
		t = []
		for word in sent:
			c = []
			for char in word:
				c.append( chardict.index(char) )
			t.append(c)
		T.append(t)
	T = np.array(T)	
	return T

def padList(sent_contents, sentMax, pad_symbol= '<pad>'):	 
	T = []
	for sent in sent_contents:
		t = []
		lenth = len(sent)
		for i in range(lenth):
			t.append(sent[i])
		for i in range(lenth, sentMax):
			t.append(pad_symbol)
		T.append(t)	
	return T
	
def findSentLengths(tr_te_list):
	lis = []
	for lists in tr_te_list:
		lis.append([len(l) for l in lists] )
	return lis


def readWordEmb(word_dict, fname, embSize=50):
	print ("Reading word vectors")
	wv = []
	wl = []
	with open(fname, 'r') as f:
		for line in f :			
			vs = line.split()
#			print (len(vs))
			if len(vs) != embSize + 1 :
				continue
			vect = list(map(float, vs[1:]))
#			print (vect[0:5] )
			wv.append(vect)
			wl.append(vs[0])
#	print ("wv",wv[0:10])
	wordemb = []
	count = 0
	oov = []
	for word in word_dict:
		if word in wl:
			wordemb.append(wv[wl.index(word)])
		else:
			count += 1
			oov.append(word)
			#wordemb.append(np.random.rand(embSize))
			wordemb.append( np.random.uniform(-np.sqrt(3.0/embSize), np.sqrt(3.0/embSize) , embSize) )
#	print (wordemb)
	#wordemb = np.asarray(map(float, wordemb))
	wordemb[word_dict.index('<pad>')] = np.zeros(embSize)
	wordemb = np.asarray(wordemb, dtype='float32')
	print ("number of unknown word in word embedding", count)
	return wordemb,oov

#Charecter level word padding
"""	
	Return:
	sentW = (s,w,c)
	sentL = (s,len(w))
"""
def padChar(data, sentMax, charMax, pad_symbol= '<pad>'):
#	print data[0]
	chardict = list ( set([char for sent in data for word in sent for char in word]) )
	tmp = [pad_symbol for i in range(charMax)]
	sentW = []
	sentL = []
	for sent in data:
		w = []
		c = []
		for x in range(len(sent)):
			word = sent[x]
			charlist = []			
			for i in range(len(word)):
				charlist.append( word[i] )
			for i in range(len(word), charMax):
				charlist.append(pad_symbol)
			w.append(charlist)
			c.append(len(word))

		for x in range(len(sent), sentMax):
			w.append(tmp)
			c.append(0)
		
		sentW.append(w)
		sentL.append(c)

	return sentW, np.array(sentL)

 	
def replaceRearWord(wl, minfreq, rear_symbol='<rear>'):
	wdict = {}
	for sent in sum(wl, []) :
		for word in sent:
			if word in wdict:
				wdict[word] += 1
			else:
				wdict[word] = 1

	wl_return = []
	for data in wl:
		t = []
		for sent in data:
			stmp = []
			for word in sent:
				if wdict[word] < minfreq :
					stmp.append(rear_symbol)
				else:
					stmp.append(word)
			t.append(stmp)
		wl_return.append(t)
 	
	if len(wl) != len(wl_return):
		print 'Mistake'
		sys.exit()
	
	return wl_return
	
	
def makeWindow(input_list, len_window, s_ind, e_ind): 	# NxM
	output_list = []
	m = len_window/2
	for sent in input_list:
		sent_new = [s_ind]*m + list(sent) + [e_ind]*m
		window = []
		for w in range(len(sent)):
			window.append( sent_new[w:w+len_window] )
		output_list.append(window)
	output_list = np.array(output_list)
	return output_list	# NxMxW

# check
#y = np.array( [[3,4,5],[1,2,3]] )
#print makeWindow(y, 5)

"""
def makeWindowChar(input_list, len_window) 	# NxMxC
	output_list = []
	m = len_window/2
	for sent in input_list:
		sent_new = 

	return output_list	# NxMxWxC
"""






