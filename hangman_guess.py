import numpy as np
import re
import collections
import itertools
import time
from collections import defaultdict
from tqdm import tqdm

clip_max = 10000

def change_char(s, p, r):
	return s[:p]+r+s[p+1:]

def num_chars(word, char):
	n = 0
	for letter in word:
		if letter == char:
			n+=1

	return n


class HangmanSolver:
	def __init__(self, dict_path, max_freq_letters=5, max_reps = 3):
		text_file = open(dict_path,"r")
		self.full_dictionary = text_file.read().splitlines()
		text_file.close()
		self.n = len(self.full_dictionary)
		self.max_freq_letters = max_freq_letters
		self.max_reps = max_reps

		new_dictionary = []
		for dict_word in self.full_dictionary:
			new_dictionary.append(''.join(set(dict_word)))

		current_dictionary_concat = ''.join(new_dictionary)

		c = collections.Counter(current_dictionary_concat)

		sorted_letter_count = c.most_common()

		self.most_freq_letters = [letter for letter, count in sorted_letter_count]

		# Hyperparameters:

		# self.weight_2 = 0.36
		# self.weight_3 = 0.14
		# self.weight_4 = 0.14
		# self.weight_5 = 0.36

		self.weight_2 = 0.08
		self.weight_3 = 0.42
		self.weight_4 = 0.25
		self.weight_5 = 0.25

		self.train()


	def start_game(self):
		mystery_index = np.random.randint(0, self.n)

		self.mystery_word = self.full_dictionary[mystery_index]

		# self.mystery_word = 'aarau'
		# self.mystery_word = 'tirunelveli'
		# self.mystery_word = 'abdominohysterectomy'
		# self.mystery_word = 'petitmaitre'

		# print('Mystery word is ' + self.mystery_word)

		self.guessed_letters = []

		a =  ord('a')

		self.unguessed_letters = [chr(i) for i in range(a,a+26)]

		self.word_len = len(self.mystery_word)

		self.current_clue = '_'*self.word_len

		self.current_dictionary = []
		for dict_word in self.full_dictionary:
			if len(dict_word) != self.word_len:
				continue
			self.current_dictionary.append(dict_word)

	def guess_v1(self):

		new_dictionary = []
		for dict_word in self.current_dictionary:
			if re.match(self.current_clue.replace("_","."), dict_word):
				new_dictionary.append(dict_word)

		self.current_dictionary = new_dictionary

		empty_chars = []
		i = 0
		for chars in self.current_clue:
			if chars == '_':
				empty_chars.append(i)
			i+=1

		new_dictionary = []
		for dict_word in self.current_dictionary:
			new_dictionary.append(''.join(set(dict_word)))

		current_dictionary_concat = ''.join(new_dictionary)

		c = collections.Counter(current_dictionary_concat)

		sorted_letter_count = c.most_common()

		num_letters = 0
		freq_chars = []
		for letter, count in sorted_letter_count:
			if letter in self.unguessed_letters:
				freq_chars.append(letter)
				num_letters += 1
				if(num_letters >= self.max_freq_letters):
					break

		IG = np.zeros(len(freq_chars))
		i = 0
		for char in freq_chars:
			# Consider every permutation of replacing '_' with char
			for num_reps in range(1, min(len(empty_chars)+1, self.max_reps)):
				for locs in itertools.combinations(empty_chars, num_reps):
					potential_word = self.current_clue
					for loc in locs:
						potential_word = change_char(potential_word, loc, char)

					new_dictionary = []
					for dict_word in self.current_dictionary:
						if re.match(potential_word.replace("_","."), dict_word):
							new_dictionary.append(dict_word)

					prob_case = len(new_dictionary)/len(self.current_dictionary)

					if prob_case !=0:
						IG[i] -= prob_case*np.log(prob_case)

			potential_word = self.current_clue

			new_dictionary = []
			for dict_word in self.current_dictionary:
				if char not in dict_word:
					new_dictionary.append(dict_word)

			prob_fail = len(new_dictionary)/len(self.current_dictionary)

			if prob_fail!=0:
				IG[i]/=prob_fail
			else:
				IG[i] = clip_max
			i+=1

		guess_index = np.argmax(IG)
		guess_char = freq_chars[guess_index]

		self.unguessed_letters.remove(guess_char)

		return guess_char


	def guess_v2(self):

		new_dictionary = []
		for dict_word in self.current_dictionary:
			if re.match(self.current_clue.replace("_","."), dict_word):
				new_dictionary.append(dict_word)

		self.current_dictionary = new_dictionary

		empty_chars = []
		i = 0
		for chars in self.current_clue:
			if chars == '_':
				empty_chars.append(i)
			i+=1

		new_dictionary = []
		for dict_word in self.current_dictionary:
			new_dictionary.append(''.join(set(dict_word)))

		current_dictionary_concat = ''.join(new_dictionary)

		c = collections.Counter(current_dictionary_concat)

		sorted_letter_count = c.most_common()

		num_letters = 0
		freq_chars = []
		for letter, count in sorted_letter_count:
			if letter in self.unguessed_letters:
				freq_chars.append(letter)
				num_letters += 1
				if(num_letters >= self.max_freq_letters):
					break

		print(freq_chars)

		IG = np.zeros(len(freq_chars))
		i = 0
		for char in freq_chars:
			# Consider every permutation of replacing '_' with char
			for num_reps in range(1, min(len(empty_chars)+1, self.max_reps)):
				for locs in itertools.combinations(empty_chars, num_reps):
					potential_word = self.current_clue
					for loc in locs:
						potential_word = change_char(potential_word, loc, char)

					new_dictionary = []
					for dict_word in self.current_dictionary:
						if re.match(potential_word.replace("_","."), dict_word) and num_chars(dict_word, char) == num_reps:
							new_dictionary.append(dict_word)

					prob_case = len(new_dictionary)/len(self.current_dictionary)

					if prob_case !=0:
						IG[i] -= prob_case*np.log(prob_case)

			potential_word = self.current_clue

			new_dictionary = []
			for dict_word in self.current_dictionary:
				if char not in dict_word:
					new_dictionary.append(dict_word)

			prob_fail = len(new_dictionary)/len(self.current_dictionary)

			if prob_fail!=0:
				IG[i]/=prob_fail
			else:
				IG[i] = clip_max
			i+=1

		guess_index = np.argmax(IG)
		guess_char = freq_chars[guess_index]

		self.unguessed_letters.remove(guess_char)

		return guess_char

	
	def train(self):
		print('Training starts.')
		model5 = defaultdict(lambda : defaultdict(lambda : defaultdict (lambda : defaultdict(lambda : defaultdict(int)))))
		model4 = defaultdict(lambda : defaultdict (lambda : defaultdict(lambda : defaultdict(int))))
		model3 = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
		model2 = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
		model1 = defaultdict(lambda : defaultdict(int))
		for word in self.full_dictionary:
			try:
				model2[len(word)][word[0]][word[1]]+=1
			except:
				continue
			model1[len(word)][word[0]]+=1
			for i in range(1,len(word)-3) :
				try :
					model3[word[i-1]][word[i]][word[i+1]]+=1
					model2[len(word)][word[i]][word[i+1]]+=1
					model1[len(word)][word[i]]+=1
					model4[word[i-1]][word[i]][word[i+1]][word[i+2]]+=1
					model5[word[i-1]][word[i]][word[i+1]][word[i+2]][word[i+3]]+=1
				except :
					continue
			try:
				i=len(word)-3
				model1[len(word)][word[i]]+=1
				model1[len(word)][word[i+1]]+=1
				model1[len(word)][word[i+2]]+=1
				model2[len(word)][word[i]][word[i+1]]+=1
				model2[len(word)][word[i+1]][word[i+2]]+=1
				model3[word[i-1]][word[i]][word[i+1]]+=1
				model3[word[i]][word[i+1]][word[i+2]]+=1
				model4[word[i-1]][word[i]][word[i+1]][word[i+2]]+=1
			except:
				continue

		self.model1 = model1
		self.model2 = model2
		self.model3 = model3
		self.model4 = model4
		self.model5 = model5

		print('Training Complete.')

	def unigram(self):
		max_prob = 0
		guess_char = ''
		for char in self.unguessed_letters:
			if self.model1[len(self.current_clue)][char] > max_prob:
				max_prob = self.model1[len(self.current_clue)][char]
				guess_char = char
		if guess_char != '':
			return guess_char
		else:
			for char in self.most_freq_letters:
				if char in self.unguessed_letters:
					return char


	def bigram(self, prior):
		assert len(prior) == len(self.unguessed_letters)
		posterior = np.zeros(len(self.unguessed_letters))

		first_empty = []
		second_empty = []

		for i in range(1,len(self.current_clue)):
			if self.current_clue[i-1] == '_' and self.current_clue[i] != '_':
				first_empty.append(i)

		for i in range(len(self.current_clue)-1):
			if self.current_clue[i+1] == '_' and self.current_clue[i] != '_':
				second_empty.append(i)

		for i in first_empty:
			letter = self.current_clue[i]
			count = 0

			for char in self.unguessed_letters:
				count+=self.model2[len(self.current_clue)][char][letter]

			if count == 0:
				count = 1
			
			for i, char in enumerate(self.unguessed_letters):
				posterior[i] += self.model2[len(self.current_clue)][char][letter]/count

		for i in second_empty:
			letter = self.current_clue[i]
			count = 0

			for char in self.unguessed_letters:
				count+=self.model2[len(self.current_clue)][letter][char]

			if count == 0:
				count = 1

			for i, char in enumerate(self.unguessed_letters):
				posterior[i] += self.model2[len(self.current_clue)][letter][char]/count
			

		posterior = posterior*self.weight_2 + prior

		max_prob = 0
		guess_char = ''
		for i, char in enumerate(self.unguessed_letters):
			if posterior[i]>max_prob:
				max_prob = posterior[i]
				guess_char = char

		if max_prob>0:
			return guess_char
		else:
			return self.unigram()

	def trigram(self, prior):
		assert len(prior) == len(self.unguessed_letters)
		posterior = np.zeros(len(self.unguessed_letters))

		first_empty = []
		second_empty = []
		third_empty = []

		for i in range(2,len(self.current_clue)):
			if self.current_clue[i-2] == '_' and self.current_clue[i-1] != '_' and self.current_clue[i] != '_':
				first_empty.append(i)

		for i in range(1,len(self.current_clue)-1):
			if self.current_clue[i-1] != '_' and self.current_clue[i] == '_' and self.current_clue[i+1] != '_':
				second_empty.append(i)

		for i in range(len(self.current_clue)-2):
			if self.current_clue[i] != '_' and self.current_clue[i+1] != '_' and self.current_clue[i+2] == '_':
				third_empty.append(i)


		for i in first_empty:
			second_letter = self.current_clue[i-1]
			third_letter = self.current_clue[i]

			count = 0
			for char in self.unguessed_letters:
				count+=self.model3[char][second_letter][third_letter]

			if count == 0:
				count = 1

			for i, char in enumerate(self.unguessed_letters):
				posterior[i] += self.model3[char][second_letter][third_letter]/count

		for i in second_empty:
			first_letter = self.current_clue[i-1]
			third_letter = self.current_clue[i+1]

			count = 0
			for char in self.unguessed_letters:
				count+=self.model3[first_letter][char][third_letter]

			if count == 0:
				count = 1

			for i, char in enumerate(self.unguessed_letters):
				posterior[i] += self.model3[first_letter][char][third_letter]/count

		for i in third_empty:
			first_letter = self.current_clue[i]
			second_letter = self.current_clue[i+1]

			count = 0
			for char in self.unguessed_letters:
				count+=self.model3[first_letter][second_letter][char]

			if count == 0:
				count = 1

			for i, char in enumerate(self.unguessed_letters):
				posterior[i] += self.model3[first_letter][second_letter][char]/count

		posterior = posterior*self.weight_3 + prior

		return self.bigram(posterior)


	def fourgram(self, prior):
		assert len(prior) == len(self.unguessed_letters)
		posterior = np.zeros(len(self.unguessed_letters))

		first_empty = []
		second_empty = []
		third_empty = []
		fourth_empty = []

		for i in range(3,len(self.current_clue)):
			if self.current_clue[i-3] == '_' and self.current_clue[i-2] != '_' and self.current_clue[i-1] != '_' and self.current_clue[i] != '_':
				first_empty.append(i)

		for i in range(2,len(self.current_clue)-1):
			if self.current_clue[i-2] != '_' and self.current_clue[i-1] == '_' and self.current_clue[i] != '_' and self.current_clue[i+1] != '_':
				second_empty.append(i)

		for i in range(1,len(self.current_clue)-2):
			if self.current_clue[i-1] != '_' and self.current_clue[i] != '_' and self.current_clue[i+1] == '_' and self.current_clue[i+2] != '_':
				third_empty.append(i)

		for i in range(len(self.current_clue)-3):
			if self.current_clue[i-2] != '_' and self.current_clue[i-1] != '_' and self.current_clue[i] != '_' and self.current_clue[i+1] == '_':
				fourth_empty.append(i)

		for i in first_empty:
			second_letter = self.current_clue[i-2]
			third_letter = self.current_clue[i-1]
			fourth_letter = self.current_clue[i]

			count = 0
			for first_letter in self.unguessed_letters:
				count+=self.model4[first_letter][second_letter][third_letter][fourth_letter]

			if count == 0:
				count = 1

			for i, first_letter in enumerate(self.unguessed_letters):
				posterior[i] += self.model4[first_letter][second_letter][third_letter][fourth_letter]/count

		for i in second_empty:
			first_letter = self.current_clue[i-2]
			third_letter = self.current_clue[i]
			fourth_letter = self.current_clue[i+1]

			count = 0
			for second_letter in self.unguessed_letters:
				count+=self.model4[first_letter][second_letter][third_letter][fourth_letter]

			if count == 0:
				count = 1

			for i, second_letter in enumerate(self.unguessed_letters):
				posterior[i] += self.model4[first_letter][second_letter][third_letter][fourth_letter]/count

		for i in third_empty:
			first_letter = self.current_clue[i-1]
			second_letter = self.current_clue[i]
			fourth_letter = self.current_clue[i+2]

			count = 0
			for third_letter in self.unguessed_letters:
				count+=self.model4[first_letter][second_letter][third_letter][fourth_letter]

			if count == 0:
				count = 1

			for i, third_letter in enumerate(self.unguessed_letters):
				posterior[i] += self.model4[first_letter][second_letter][third_letter][fourth_letter]/count

		for i in fourth_empty:
			first_letter = self.current_clue[i-2]
			second_letter = self.current_clue[i-1]
			third_letter = self.current_clue[i]

			count = 0
			for fourth_letter in self.unguessed_letters:
				count+=self.model4[first_letter][second_letter][third_letter][fourth_letter]

			if count == 0:
				count = 1

			for i, fourth_letter in enumerate(self.unguessed_letters):
				posterior[i] += self.model4[first_letter][second_letter][third_letter][fourth_letter]/count

		posterior = posterior*self.weight_4 + prior

		return self.trigram(posterior)


	def fivegram(self):
		posterior = np.zeros(len(self.unguessed_letters))

		first_empty = []
		second_empty = []
		third_empty = []
		fourth_empty = []
		fifth_empty = []

		for i in range(4,len(self.current_clue)):
			if self.current_clue[i-4] == '_' and self.current_clue[i-3] != '_' and self.current_clue[i-2] != '_' and self.current_clue[i-1] != '_' and self.current_clue[i] != '_':
				first_empty.append(i)

		for i in range(3,len(self.current_clue)-1):
			if self.current_clue[i-3] != '_' and self.current_clue[i-2] == '_' and self.current_clue[i-1] != '_' and self.current_clue[i] != '_' and self.current_clue[i+1] != '_':
				second_empty.append(i)

		for i in range(2,len(self.current_clue)-2):
			if self.current_clue[i-2] != '_' and self.current_clue[i-1] != '_' and self.current_clue[i] == '_' and self.current_clue[i+1] != '_' and self.current_clue[i+2] != '_':
				third_empty.append(i)

		for i in range(1,len(self.current_clue)-3):
			if self.current_clue[i-1] != '_' and self.current_clue[i] != '_' and self.current_clue[i+1] != '_' and self.current_clue[i+2] == '_' and self.current_clue[i+3] != '_':
				fourth_empty.append(i)

		for i in range(len(self.current_clue)-4):
			if self.current_clue[i] != '_' and self.current_clue[i+1] != '_' and self.current_clue[i+2] != '_' and self.current_clue[i+3] != '_' and self.current_clue[i+4] == '_':
				fifth_empty.append(i)

		for i in first_empty:
			second_letter = self.current_clue[i-3]
			third_letter = self.current_clue[i-2]
			fourth_letter = self.current_clue[i-1]
			fifth_letter = self.current_clue[i]

			count = 0
			for first_letter in self.unguessed_letters:
				count+=self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]

			if count == 0:
				count = 1

			for i, first_letter in enumerate(self.unguessed_letters):
				posterior[i] += self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]/count

		for i in second_empty:
			first_letter = self.current_clue[i-3]
			third_letter = self.current_clue[i-1]
			fourth_letter = self.current_clue[i]
			fifth_letter = self.current_clue[i+1]

			count = 0
			for second_letter in self.unguessed_letters:
				count+=self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]

			if count == 0:
				count = 1

			for i, second_letter in enumerate(self.unguessed_letters):
				posterior[i] += self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]/count

		for i in third_empty:
			first_letter = self.current_clue[i-2]
			second_letter = self.current_clue[i-1]
			fourth_letter = self.current_clue[i+1]
			fifth_letter = self.current_clue[i+2]

			count = 0
			for third_letter in self.unguessed_letters:
				count+=self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]

			if count == 0:
				count = 1

			for i, third_letter in enumerate(self.unguessed_letters):
				posterior[i] += self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]/count

		for i in fourth_empty:
			first_letter = self.current_clue[i-1]
			second_letter = self.current_clue[i]
			third_letter = self.current_clue[i+1]
			fifth_letter = self.current_clue[i+3]

			count = 0
			for fourth_letter in self.unguessed_letters:
				count+=self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]

			if count == 0:
				count = 1

			for i, fourth_letter in enumerate(self.unguessed_letters):
				posterior[i] += self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]/count

		for i in fifth_empty:
			first_letter = self.current_clue[i]
			second_letter = self.current_clue[i+1]
			third_letter = self.current_clue[i+2]
			fourth_letter = self.current_clue[i+3]

			count = 0
			for fifth_letter in self.unguessed_letters:
				count+=self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]

			if count == 0:
				count = 1

			for i, fifth_letter in enumerate(self.unguessed_letters):
				posterior[i] += self.model5[first_letter][second_letter][third_letter][fourth_letter][fifth_letter]/count

		posterior = posterior*self.weight_5

		return self.fourgram(posterior)


	def guess_ngram(self):

		guess_char = self.fivegram()
		self.unguessed_letters.remove(guess_char)
		return guess_char

	def step(self, guess):
		if guess in self.mystery_word:
			indices = []
			for i in range(len(self.mystery_word)):
				if  self.mystery_word[i] == guess:
					indices.append(i)

			for i in indices:
				self.current_clue = change_char(self.current_clue, i, guess)

			return 0

		else:
			return -1

	def tune_hyperparameters(self, batch_size, num_batches, T0, delta0):
		performance = 0.0
		T = T0
		delta = delta0
		for batch in range(num_batches):
			N_success = 0.0
			for count in range(batch_size):
				hangman_agent.start_game()
				N_left = N_strikes
				while N_left>0:
					# guess = hangman_agent.guess()
					guess = hangman_agent.guess_ngram()
					N_left += hangman_agent.step(guess)
					if('_' in hangman_agent.current_clue):
						continue
					else:
						N_success += 1
						break

			DE = N_success - performance

			if N_success>performance:
				performance = N_success

			print(batch, self.weight_2, self.weight_3, self.weight_4, self.weight_5)

			print('Accuracy of the algorithm: ', str(N_success/batch_size * 100))

			p_sample = np.exp(-max(DE,0)/ T)

			if np.random.uniform(0,1)<p_sample:
				x = np.random.uniform(-delta, delta, 5)
				x[4] = 0
				x[0] = 0

				self.weight_2 += x[1] - x[0]
				self.weight_3 += x[2] - x[1]
				self.weight_4 += x[3] - x[2]
				self.weight_5 += x[4] - x[3]

			T/=1.1
			delta/=1.05




dictionary_path = './words_250000_train.txt'

N_strikes = 6
N_samples = 100

N_success = 0.0

hangman_agent = HangmanSolver(dictionary_path)

hangman_agent.tune_hyperparameters(100, 100, 100, 0.2)

# 0.03864019512995809 0.15428121558276808 0.12841833151811646 0.6786602577691578

for count in range(N_samples):
	a = time.time()
	print(count)
	hangman_agent.start_game()
	N_left = N_strikes
	while N_left>0:
		# guess = hangman_agent.guess()
		guess = hangman_agent.guess_ngram()
		print('current clue: ' + hangman_agent.current_clue)
		N_left += hangman_agent.step(guess)
		print('guess letter: ' + guess)
		if('_' in hangman_agent.current_clue):
			continue
		else:
			print('Sucess! The mystery word was: ' + hangman_agent.mystery_word)
			print('Number of strikes used: ' + str(N_strikes - N_left))
			N_success += 1
			break

	print('Done in ' + str(time.time()-a) + ' seconds')

print('Accuracy of the algorithm: ', str(N_success/N_samples * 100))