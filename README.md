## Part-of-Speech Tagging
A basic problems in Natural Language Processing is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech.

Part of speech tagging is prime, and there are multiple ways to tag words in a given sentence to its correspoding tag. To solve this age-old problem, I have implemented three different techniques to do the task. The three methods showcase different probabilistic methods, that build-up on one another.

1) Naive Bayes - Simplified
2) Viterbi Algorithm - HMM
3) MCMC

Essentially in our training data, we have a set of 12 POS tags that are defined - 
("adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", "verb", 'x', ".").


## Approach Taken

For the given training data that includes sentences and their corresponding tags, it is pivotal for us to formulate emission and transmission probabilities. These will be vital for the above algorithms mentioned. 

1) States
("adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", "verb", 'x', ".")

2) Initial Probabilty
This is the probability of occurence of a given POS tag based on the train distribution.

3) Emission Probability
The probabilities are stored in a dictionary, where POS tags behave as the keys. 
Eg - emit_p['noun'] = {'alarm': 0.4, 'keynotes': 0.3, 'geddes': 0.1, 'matri': 0.2}

4) Transmission Probability
These probabilities are vital for the Viterbi and MCMC algorithm. They are also stored in a similar fashion as mentioned above.
Eg -   trans_p['noun'] = {'adj': 0.013302810632074271,
 'adv': 0.02500085785479103, 'adp': 0.263381658513818, 'conj': 0.0842948174505748, 'det': 0.024047446418438107,'noun': 0.24658044256491474,'num': 0.019190676468607636,'pron': 0.039503716718655046, 'prt': 0.03877068711847653,'verb': 0.35950755386491084,'x': 0.001302061122088027,'.': 0.9999828995526692}

## Code Summary

###   Naive Bayes 

Naive Bayes considers each POS tag independent of one another. Using Naive Bayes Inference, POS tags are determined for the observed set of states.

The formulation of the Bayesian net is as follow: 

** Simplified Bayes net ** 
Naive Bayes  : P(POS/W)=P(W/POS).P(POS)/P(W)  
The denomenator is useless or irrelevant for us.

Based on the tag with the highest(max) probability, the tag sequence is generated. Well now we now why it is called 'Naive'.

### Viterbi Algorithm - HMM

Viterbi algorithm is based on dynamic programming. Previously computed probabilities are stored and can be used to intermediate calculations and need not be computed over and over again.


Viterbi can be used to find the highest probable tag sequence in an effecient and recursive manner. I have implemented a bigram HMM, although a trigram HMM would perform better.  Since there are no emission probabilities for new words, we tried several different types of fixes for this problem. Assigning a fractional probability worked out, and this is the design decision which worked out for us. 

### MCMC - Gibbs

Created a dictionary of word and POS pairs, and computed the corresponding frequency. 

Steps :
For n iterations, the POS tag was computed using a weighted probability distribution, for a given word in the sentence. 
The POS tag is computed based on the probability distribution computed using the dictionary of word and POS tags.
The frequencies are updated in each iteration, and a new POS tag is sampled each iteration.
Once the iterations are complete, the POS tag with the highest frequency is chosen as the final POS tag.

## Results

==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
   1. Simple:             93.92%               47.45%
   2. HMM:                95.03%               54.15%
   3. Complex:            90.80%               33.90%
