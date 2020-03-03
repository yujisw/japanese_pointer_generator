import numpy as np

vocab_path = '../data/vocab'
with open(vocab_path,'r') as f:
    vocab = f.read().split('\n')
vocab = [row.split(' ')[0] for row in vocab]

path = '../jawiki.word_vectors.200d.txt'
with open(path,'r') as f:
    n,d = f.readline().split(' ')
    vecs = None
    words = []
    for i in range(int(n)):
        row = f.readline().split(' ')
        word = row[0]
        if word in vocab:
            words.append(word)
            vec = np.array([float(v) for v in row[1:]])
            if vecs is None:
                vecs = vec.reshape(1,-1)
            else:
                vecs = np.insert(vecs, len(vecs), vec, axis=0)
            # vecs[i] = vec
            print(word)
            print(vecs.shape)
print(vecs.shape)
print(len(words))

pretrained_vocab_path = '../data/pretrained_vocab'
with open(pretrained_vocab_path,'w') as vocab:
    vocab.write('\n'.join(words))

pretrained_vec_path = '../data/pretrained_vec'
np.save(pretrained_vec_path,vecs)