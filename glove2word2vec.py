'''
    This script converts a GLOVE word embedding into a conventional textual format
'''
import sys
import gensim

glove_input = ""

if(len(sys.argv) < 2):
    print("You must provide the input and the ouput file. e.g. python glove2gensim.py path/to/glove.txt path/to/newgensim.txt")
    exit(1)

glove_input = sys.argv[1]
gensim_output = sys.argv[2]
vector_size = 0
vocab_size = 0

with open(glove_input, 'r') as f:
    vector_size = len(f.readline().split())-1 
    vocab_size = 1 + sum(1 for k in f) # Since one line is read out of the file's buffer we have to add one extra line

with open(glove_input, 'r') as f_in:
    with open(gensim_output, 'w') as f_out:
        f_out.write(str(vocab_size) + ' ' + str(vector_size) + '\n')
        for line in f_in:
            f_out.write(line)

model=gensim.models.KeyedVectors.load_word2vec_format(gensim_output,binary=False) #GloVe Model
print(model.most_similar(positive=['creep'], topn=10))
print(model.similarity('king', 'queen'))
