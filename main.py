import numpy as np
import pandas as pd
from scipy import optimize
import pandas as pd
import random

#helper functions

def likelihood(M,ps,pu):
	'''Computes the likelihood that you have seen M words given arrays 
	representing words you do and don't know. ps, pu are np arrays of the 
	frequencies of the words you do and don't know, respectively.'''
    
	return np.prod(1-(1-ps)**M)*np.prod((1-pu)**M)

def max_likelihood(ps,pu):
	'''Computes the maximum likelihood value of M = number of words seen
	givenn ps, pu arrays.

	Currently just brute forcing it.'''
    
	return np.argmax(np.array(  [likelihood(m,ps,pu) for m in range(0,50000000,10000)] ))

def scipy_max_likelihood(ps,pu):
	'''Compute ML estimate of M using scipy optimization function.'''

	f = lambda x: -likelihood(x,ps,pu)

	init_guess = 1 / (np.min(ps))

	return optimize.fmin(f,1)[0]

def new_word_prob(pi,m):
	'''Probability that the mth word you see is new to you,
	given a frequency table pi.'''
    
	return np.sum(pi*((1-pi)**m))

class VocabSizeEstimator:
	'''Class to extimate your vocab size based on words you do and don't know and 
	their frequency in a self.corpus.'''

	def __init__(self,corpus_path):
		'''self.Corpus should be a csv file with a list of word, relative frequency.'''

		#load in corpus
		self.corpus = pd.read_csv(corpus_path, header=False)
		self.corpus.columns = ['word','freq']

		#use corpus to calculate words seen - words known conversion
		#need to approximate to get to big enough numbers
		probs = np.array(self.corpus['freq'])

		new_words = np.array([ new_word_prob(probs,m) for m in range(10000) ])
		sampled_new_words = np.array([ new_word_prob(probs,m) for m in range(0,50000000,10000) ])

		self.words_known_sampled = sampled_new_words.cumsum()*10000 + new_words.sum()

	def estimate_words_seen(self,sample,known):
		'''Get estimate of total words seen, given a sample of words (dataframe) and
		a boolean array of whether you know them.'''

		sample_p = list(sample['freq'])
		s = np.array([x for x,y in zip(sample_p,known) if y])
		u = np.array([x for x,y in zip(sample_p,known) if not y])

		return int(scipy_max_likelihood(s,u))

	def estimate_words_known_corpus(self, words_seen):
		'''Now we need to go from words seen to words known, i.e. *unique* words seen.
		Here we base it on the frequencies of words in the self.corpus. Note that this assumes
		there are no other words, so the estimate is bounded by the size of the self.corpus freq table.'''

		words_seen_nearest_10k = int(words_seen / 10000)
		return int(self.words_known_sampled[words_seen_nearest_10k])

	def generate_sample(self,size,rare_threshold):
		'''Generate a sample of rare words from the self.corpus list.'''

		rare_words = self.corpus[self.corpus['freq']<rare_threshold]
		rows = random.sample(rare_words.index,size)
		return rare_words.loc[rows]



if __name__=='__main__':

	s = np.array([0.000001,0.001,0.0004,0.00001])
	u = np.array([0.001])

	print(max_likelihood(s,u))
	print()
	print(scipy_max_likelihood(s,u))

