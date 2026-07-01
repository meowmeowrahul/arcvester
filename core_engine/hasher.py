import numpy as np 
# Rturns A set of shingles 
#   def shingle(text:string,k:int):
#       shingle_set=[]
#       for i in range(len(text)-k+1):
#           shingle_set.append(text[i:k+i])
#       return set(shingle_set)
#   def vocab(shingle_sets:list):
#       full_set={item for set_ in shingle_sets for item in set_}
#       vocab = {}
#       for i,shingle in enumerate(list(full_set)):
#           vocab[shingle] = i 
#       return vocab
#   def one_hot_encoder(shingles:set,vocab:dict):
#       vec = np.zeros((len(vocab)))
#       for shingle in shingles:
#           index = vocab[shingle]
#           vec[index] = 1
#       return vec
#   def min_hash(vocab,resolution):
#       length = len(vocab)
#       arr = np.zeros((resolution,length))
#       for i in range(resolution):
#           permutation = np.random.permutation(len(vocab))+1
#           arr[i,:]= permutation.copy()
#       return arr
def get_signature(hyperplanes,vector):
    projections = np.dot(hyperplanes,vector)
    return (projections > 0).astype(int)

