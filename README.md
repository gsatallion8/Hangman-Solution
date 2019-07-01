# Hangman-Solution
An agent to play the game of Hangman in both information retrieval and word guessing scenarios.

## Information Retrieval:
Consider the example of {ABC, ABD, AEF, EGH} as a dictionary. Also assume only one strike is available. Going by the frequency based approach, one would wanna guess A as the first guess. The success rate of this would be (3/4)x(2/3)x(1/2) = 1/4 (A, B, C or D being guesses). Whereas, guessing E first would give us a fail rate of (1/2) and once the first guess is a success, the position of E would reveal the word. Hence E is a better guess from the perspective of information retrieval. Hence, one would be better off using Information theoretic metrics for guessing a letter. In this project, the guess_v1() and guess_v2() use entropy as a way to decide which letter must me chosen as the next guess.

## Word Guessing:
Entropy based methods only work when the mystery word is in the available dictionary. Else, one would have to resort to learninng ngram based models hoping learned ngrams reccur in the unknown words as well. In this project, guess_ngram() uses a cascade of 5, 4, 3, 2, 1 gram models as a way to find the most probable next guess ought to be. A weighted average of probailities from each of the ngram model is used to decide which letter has to be picked.
