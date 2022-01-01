###################################
# CS B551 Spring 2021, Assignment #3
#
# (Based on skeleton code by D. Crandall)
#


import random
import math



class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            p = 0
            for i in range(len(sentence)):
                try:
                    int_p = emit_p[label[i]][sentence[i]] * pos_counter_list[label[i]]/sum(pos_counter_list.values())
                    p += math.log10(int_p)
                except:
                    print(p)
                    pass
            return p
            #return -999
        elif model == "HMM":


            return -999

        elif model == "Complex":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):

        train_input = data
        adjective_words=[]
        adverb_words=[]
        adposition_words=[]
        conjunction_words=[]
        determiner_words=[]
        noun_words=[]
        number_words=[]
        pronoun_words=[]
        particle_words=[]
        verb_words=[]
        foreign_word_words=[]
        punctuation_mark_words=[]
        dummy_list=[]

        for i in range(len(train_input)):
            for x in train_input[i][1]:
                dummy_list.append(x)
                
        unique_pos=set(dummy_list)

        all_words=[]

        #Collating all words for a POS tag
        for x in train_input:
            for i in range(len(x[0])):
                #print(x[1][i])
                all_words.append(x[0][i])
                if x[1][i]=="adj":
                    adjective_words.append(x[0][i])
                elif x[1][i]=="adp":
                    adposition_words.append(x[0][i])
                elif x[1][i]=="adv":
                    adverb_words.append(x[0][i])
                elif x[1][i]=="conj":
                    conjunction_words.append(x[0][i])
                elif x[1][i]=="det":
                    determiner_words.append(x[0][i])
                elif x[1][i]=="noun":
                    noun_words.append(x[0][i]) 
                elif x[1][i]=="num":
                    number_words.append(x[0][i])         
                elif x[1][i]=="pron":
                    pronoun_words.append(x[0][i])             
                elif x[1][i]=="prt":
                    particle_words.append(x[0][i])             
                elif x[1][i]=="verb":
                    verb_words.append(x[0][i])
                elif x[1][i]=="x":
                    foreign_word_words.append(x[0][i])             
                elif x[1][i]==".":
                    punctuation_mark_words.append(x[0][i])                         

        #simplified Bayes net            
        #Naive Bayes says : P(POS/W)=P(W/POS).P(POS)/P(W)  
        #denomenator is useless

        #Need to compute count for each word for each POS
        #Creating a dictionary of Words in each POS lists

        #Getting set of each POS
        adjective_words_unique=set(adjective_words)
        adverb_words_unique=set(adverb_words)
        adposition_words_unique=set(adposition_words)
        conjunction_words_unique=set(conjunction_words)
        determiner_words_unique=set(determiner_words)
        noun_words_unique=set(noun_words)
        number_words_unique=set(number_words)
        pronoun_words_unique=set(pronoun_words)
        particle_words_unique=set(particle_words)
        verb_words_unique=set(verb_words)
        foreign_word_words_unique=set(foreign_word_words)
        punctuation_mark_words_unique=set(punctuation_mark_words)

        #creating dict with word and word counts
        adjective_words_counter={}
        adverb_words_counter={}
        adposition_words_counter={}
        conjunction_words_counter={}
        determiner_words_counter={}
        noun_words_counter={}
        number_words_counter={}
        pronoun_words_counter={}
        particle_words_counter={}
        verb_words_counter={}
        foreign_word_words_counter={}
        punctuation_mark_words_counter={}

        #Getting count
        for x in list(adjective_words_unique):
            adjective_words_counter[x]=adjective_words.count(x)    
        for x in list(adverb_words_unique):
            adverb_words_counter[x]=adverb_words.count(x)
        for x in list(adposition_words_unique):
            adposition_words_counter[x]=adposition_words.count(x)
        for x in list(conjunction_words_unique):
            conjunction_words_counter[x]=conjunction_words.count(x)
        for x in list(determiner_words_unique):
            determiner_words_counter[x]=determiner_words.count(x)
        for x in list(noun_words_unique):
            noun_words_counter[x]=noun_words.count(x)
        for x in list(number_words_unique):
            number_words_counter[x]=number_words.count(x)
        for x in list(pronoun_words_unique):
            pronoun_words_counter[x]=pronoun_words.count(x)
        for x in list(particle_words_unique):
            particle_words_counter[x]=particle_words.count(x)
        for x in list(verb_words_unique):
            verb_words_counter[x]=verb_words.count(x)
        for x in list(foreign_word_words_unique):
            foreign_word_words_counter[x]=foreign_word_words.count(x)
        for x in list(punctuation_mark_words_unique):
            punctuation_mark_words_counter[x]=punctuation_mark_words.count(x)

        #Computing prior Probability 
        total_words=adjective_words+adverb_words+adposition_words+conjunction_words+determiner_words+noun_words+number_words+pronoun_words+particle_words+verb_words+foreign_word_words+punctuation_mark_words

        adjective_probability=sum(adjective_words_counter.values())/len(total_words)
        adverb_probability=sum(adverb_words_counter.values())/len(total_words)
        adposition_probability=sum(adposition_words_counter.values())/len(total_words)
        conjunction_probability=sum(conjunction_words_counter.values())/len(total_words)
        determiner_probability=sum(determiner_words_counter.values())/len(total_words)
        noun_probability=sum(noun_words_counter.values())/len(total_words)
        number_probability=sum(number_words_counter.values())/len(total_words)
        pronoun_probability=sum(pronoun_words_counter.values())/len(total_words)
        particle_probability=sum(particle_words_counter.values())/len(total_words)
        verb_probability=sum(verb_words_counter.values())/len(total_words)
        foreign_word_probability=sum(foreign_word_words_counter.values())/len(total_words)
        punctuation_mark_probability=sum(punctuation_mark_words_counter.values())/len(total_words)

        #Calculate Probability of a word having a POS tag(each)
        unique_pos=["adjective","adverb","adposition","conjunction","determiner","noun","number","pronoun","particle","verb","foreign_word","punctuation_mark"]


        final_dict={}
        check = train_input[28]
        for word in check[0][0]:
            sample_word=word
            word_pos_probability={}
            for x in unique_pos:
                if str(sample_word) in eval('%s_words_counter.keys()' %x):
                    y = eval('%s_words_counter[sample_word]'%x)/ all_words.count(str(sample_word))
                    word_pos_probability['word_being_%s'%x] = y
                else:
                    word_pos_probability['word_being_%s'%x]=0
            final_dict[sample_word]=word_pos_probability
        
        global word_counter_list
        word_counter_list = {"adj" :adjective_words_counter, 
        "adv" :adverb_words_counter,
        "adp" : adposition_words_counter,
        "conj" : conjunction_words_counter,
        "det" :determiner_words_counter,
        "noun" :noun_words_counter,
        "num" :number_words_counter,
        "pron" :pronoun_words_counter,
        "prt" :particle_words_counter,
        "verb" :verb_words_counter,
        "x" :foreign_word_words_counter,
        "." :punctuation_mark_words_counter}

        #Calculating emit probability, using the word_counter_list defined above
        global emit_p 
        emit_p = {}
        for i in word_counter_list:
            emit_p[i] = word_counter_list[i]
            emit_p[i] = {j : emit_p[i][j]/ sum(emit_p[i].values())for j in emit_p[i].keys()}

        global start_p
        start_p = {"adj": adjective_probability,
           "adv":adverb_probability,
           "adp":adposition_probability, "conj":conjunction_probability, 
           "det":determiner_probability, "noun":noun_probability,
           "num":number_probability, "pron":pronoun_probability, 
           "prt":particle_probability, "verb":verb_probability,
           'x':foreign_word_probability, ".":punctuation_mark_probability}

        # This was tricky to calculate initially
        global trans_p
        trans_p = {
            "adj": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "adp": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "adv": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "conj": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "det": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "noun": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "num": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "pron": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "prt": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "verb": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            "x": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},
            ".": {"adj": 0, "adv":0, "adp" : 0, "conj" : 0, "det" : 0, "noun": 0, "num" : 0, "pron": 0, "prt":0, "verb" : 0, "x":0, ".":0},    

        }

        for i in range(0,len(train_input)):
            for j in range(len(train_input[i][1])-1):

                trans_p[train_input[i][1][j]][train_input[i][1][j+1]]+=1
                
        for i in trans_p.keys():
            val = sum(trans_p[i].values())
            for j in trans_p[i].keys():
                trans_p[i][j] = trans_p[i][j] / val

        global states
        states = ("adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", "verb", 'x', ".")

        global pos_counter_list
        pos_counter_list = {"adj" :adjective_words, 
        "adv" :adverb_words,
        "adp" : adposition_words,
        "conj" : conjunction_words,
        "det" :determiner_words,
        "noun" :noun_words,
        "num" :number_words,
        "pron" :pronoun_words,
        "prt" :particle_words,
        "verb" :verb_words,
        "x" :foreign_word_words,
        "." :punctuation_mark_words}
        for i in pos_counter_list:
            pos_counter_list[i] = len(pos_counter_list[i])

        global final
        final = {}
        import collections
        final = collections.defaultdict(dict)
        for x in train_input:
            for i in range(len(x[0])):
                try:
                    final[x[0][i]][x[1][i]]+=1
                except:
                    final[x[0][i]][x[1][i]]=1
                    
        import copy

        


    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        input_words = sentence
        tags_list =['']*len(input_words)

        for j in range(len(input_words)):
            p = 0
            for s in states:
                tag = s

                try:
                    p1 = emit_p[tag][input_words[j]]* start_p[tag]
                except:
                    p1 = 0.000000000000001*start_p[tag]

                if p1 > p:
                    tags_list[j] = tag
                    p = p1
        return tags_list

    # Inspiration from Prof. Crandall's pseudocode shared in class/modules
    def hmm_viterbi(self, sentence):
        states = ("adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", "verb", 'x', ".")
        trans = trans_p
        emission = emit_p
        initial = start_p

        observed = list(sentence)

        N=len(observed)
        V_table = {states[i]:[0]*N for i in range(0,len(states))}
        which_table = {states[i]:[0]*N for i in range(0,len(states))}
        for s in states:
            try:
                s1 = initial[s] * emission[s][observed[0]]
            except:
                s1 = 1/10000000000000        
            V_table[s][0] = s1

        for i in range(1, N):
            for s in states:
                (which_table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] * trans[s0][s]) for s0 in states ], key=lambda l:l[1] ) 
                try:
                    V_table[s][i] *= emission[s][observed[i]]
                except:
                    V_table[s][i] *= 1/10000000000000

        viterbi_seq = [""] * N
        
        max_till_now=0
        max_index=0
        try:
            for x in states:
                if V_table[x][i]>=max_till_now:
                    max_till_now=V_table[x][i]
                    max_index=x
            viterbi_seq[N-1] = max_index
            for i in range(N-2, -1, -1):
                viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]
        except:
            viterbi_seq[0] = list(V_table.keys())[list(V_table.values()).index(max(V_table.values()))]

        return viterbi_seq

    # Discussed the approach with one of our classmates Sreesha K. 
    def complex_mcmc(self, sentence):
        draw_list = []
        tags = []
        score = []
        tags = []
        d1 = final
        from numpy.random import choice

        for word in sentence:
            if(word not in d1.keys() or d1[word]=={}):
                tags.append(choice([item[0] for item in start_p.items()], 1, p = [item[1] for item in start_p.items()]))


            else:

                l = d1[word].items()
                l1 = [item[0] for item in l]
                l2 = [item[1] for item in l]
                for i in range(0,500):
                    if(i!=0):
                        l2[l1.index(draw[0])]+=1

                    val = sum(l2)

                    l2 = [item/val for item in l2]
                    draw = choice(l1, 1, p = l2)
                tags.append(l1[l2.index(max(l2))])

        return tags

        #return [ "noun" ] * len(sentence)



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

