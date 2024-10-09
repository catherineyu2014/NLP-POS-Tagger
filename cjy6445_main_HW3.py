import re
import argparse
from collections import defaultdict
import math
    
def read_file(filename):
    with open (filename, 'r') as file:
        data = file.readlines()
    return data

def process_probabilities2(likelihood):
    total = 0
    
    for k,v in likelihood.items():
        for count in v.values():
            total += count
            
    for k,v in likelihood.items():
        for inner_key in v:
            v[inner_key] = v[inner_key]/total
            
    return likelihood

def process_probabilities(counts):
    probabilities = defaultdict(lambda: defaultdict(float))
    for tag, word_counts in counts.items():
        total = sum(word_counts.values())
        for word, count in word_counts.items():
            probabilities[tag][word] = math.log(count / total)  # Using log probabilities
    return probabilities

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
    
def handle_oov(word):
    if word.lower() == 'the' or word.lower() == 'this':
        return 'DT'
    elif word.lower() == 'to':
        return 'TO'
    elif word.istitle():
        return 'NNP'
    elif word.lower() == 'it':
        return 'PRP'
    elif word.isdigit() or word.startswith('$') or isfloat(word):
        return 'CD'
    elif word.endswith("'s"):
        return 'POS'
    elif '-' in word:
        return 'NNP'
    elif word.endswith('ing'):
        return 'VBG'
    elif word.endswith('ed'):
        return 'VBD'
    elif word.endswith('s') and len(word)>2:
        return 'NNS'
    elif word.endswith('ly'):
        return 'RB'
    elif word.endswith('est'):
        return 'JJS'
    elif word.endswith('er'):
        return 'JJR'
    elif word.endswith('able') or word.endswith('ible') or word == 'final' or word.lower() == 'big':
        return 'JJ'
    elif word == 'will' or word == 'may':
        return 'MD'
    elif word == 'be':
        return 'VB'
    elif word == 'and':
        return 'CC'
    elif word == ';' or word =='--':
        return ':'
    else:
        return 'NN'

def viterbi(words_list, tag_list, transition_probabilities, emission_probabilities):
    oov = math.log(1e-20)

    num_tags = len(tag_list)
    num_words = len(words_list)
    
    viterbi = []
    backpointer = []
    for _ in range(num_words):
        viterbi.append({})
        backpointer.append({})
        
    # initialize
    first_word = words_list[0]
    
    # loop through dictionary
    for tag in tag_list:
        emission_prob = emission_probabilities.get(tag, {}).get(first_word, oov)
        viterbi[0][tag] = emission_probabilities["Begin_Sent"].get(tag, oov)*emission_prob
        backpointer[0][tag] = 0 # no previous word at first word
    
    #print (viterbi)
    # recursion
    for i in range(1,num_words):
        curr_word = words_list[i]
        
        # handle oov
        if curr_word.lower() not in emission_probabilities[tag_list[0]]:
            estimated_tag = handle_oov(curr_word)
        else:
            estimated_tag = None

        for curr_tag in tag_list:
            max_prob = float('-inf')
            best_prev_tag = None
            
            transition_prob = transition_probabilities.get(curr_tag, {}).get(curr_word, oov)

            for prev_tag in viterbi[i-1]:
                emission_prob = emission_probabilities.get(prev_tag, {}).get(curr_tag, oov)
                prob = viterbi[i-1][prev_tag] + transition_prob + emission_prob

                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag
                    
            if max_prob > float('-inf'):
                viterbi[i][curr_tag] = max_prob
                backpointer[i][curr_tag] = best_prev_tag
        
    # find best path at the end
    final_max_prob = float('-inf')
    best_last_tag = None
    for tag in viterbi[-1]:
        if viterbi[-1][tag] > final_max_prob:
            final_max_prob = viterbi[-1][tag]
            best_last_tag = tag
    
    # backtracking
    best_tag_sequence = []

    if best_last_tag is None:
        print("Error: No valid tag found for the last word.")
    else:
        best_tag_sequence.append(best_last_tag)
        
    for i in range(num_words-1,0,-1):
        if best_tag_sequence[0] is not None and best_tag_sequence[0] in backpointer[i]:
            best_tag_sequence.insert(0, backpointer[i][best_tag_sequence[0]])
        else:
            print("backpointer none")
                    
    return best_tag_sequence

def main():
    parser = argparse.ArgumentParser(description='Process file for POS tagging')
    parser.add_argument('filename', type=str, help='The file to process')
    args = parser.parse_args()

    data = []
    data.extend(read_file('WSJ_02-21.pos'))
    data.extend(read_file('WSJ_24.pos'))
    
    # likelihood dictionary keeps track of all the POS and the counts of each POS
    likelihood = {}
    # vocabulary is a list that keeps track of all the words in the vocabulary
    vocabulary = []
    # tag list
    tag_list = []
    
    # transition dictionary keeps track of beg of sentence
    transition = {}
    transition["Begin_Sent"] = {}
    transition["End_Sent"] = {}

    # prev is set to the one before the first item
    prev = ""
    
    # iterate through data and add to lists
    for i in range(len(data)):
        d = data[i]

        # extract POS and word itself
        # at end of sentence. add prev to transition
        if d == '\n':
            if prev not in transition["End_Sent"]:
                transition["End_Sent"][prev] = 1
            else:
                transition["End_Sent"][prev] += 1
            prev = '\n'
            continue
        elif d == '':
            continue
        else:
            split = d.strip("\n")
            split = split.split("\t")
            word = split[0].lower()
            pos = split[1]
            
            # middle of sentence
            if prev not in transition:
                transition[prev] = {}
            if pos not in transition[prev]:
                transition[prev][pos] = 1
            else:
                transition[prev][pos] += 1
                
            # beg of file
            if i==0:
                transition["Begin_Sent"][pos] = 1
                
            # beg of sentence
            if prev == "\n":
                if pos not in transition["Begin_Sent"]:
                    transition["Begin_Sent"][pos] = 1
                else:
                    transition["Begin_Sent"][pos] += 1

        # add to dict
        if pos not in likelihood:
            likelihood[pos] = {}
            tag_list.append(pos)

        if word in likelihood[pos]:
            likelihood[pos][word] += 1
        else:
            likelihood[pos][word] = 1
            vocabulary.append(word.lower())
        
        # update prev variable for next iteration
        prev = pos

    transition_probabilities = process_probabilities(likelihood)
    emission_probabilities = process_probabilities(transition)
    
    
    # get list of words for input & strip
    words_list_all = read_file(args.filename)
    words_list = []
    
    for word in words_list_all:
        words_list.append(word.strip('\n'))
    
    tag_sequence = viterbi(words_list, tag_list, transition_probabilities, emission_probabilities)

    # create output file and print out
    file = open('submission.pos', 'w')
    for i in range(len(words_list)):
        if words_list[i]=="":
            file.write('\n')
            continue
        file.write(words_list[i]+'\t'+tag_sequence[i]+'\n')
    
if __name__ == '__main__':
    main()

