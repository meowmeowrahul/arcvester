import numpy as np 
from itertools import combinations
import torch
class LSH:
    def __init__(self,b):
        self.buckets = []
        self.counter = 0
        self.b = b
        for i in range(b):
            self.buckets.append({})
    def cosine_similarity(self,vector_A, vector_B):
        return (np.dot(vector_A, vector_B)) / (np.linalg.norm(vector_A) * np.linalg.norm(vector_B)) 
        
    def make_subvecs(self,signature):
        l = len(signature)
        assert l%self.b==0
        r = int(l/self.b)
        subvec = []
        for i in range(0,l,r):
            subvec.append(signature[i:i+r])
        return np.stack(subvec)
    def add_hash(self,signature):
        subvecs = self.make_subvecs(signature).astype(str)
        for i,subvec in enumerate(subvecs):
            subvec = ','.join(subvec)
            if subvec not in self.buckets[i].keys():
                self.buckets[i][subvec]=[]
            self.buckets[i][subvec].append(self.counter)
        self.counter +=1
    def check_candidates(self, ids):
        candidates = set()
        
        # Iterate through each band (bucket dictionary)
        for band in self.buckets:
            # Look at every bucket in the current band
            for bucket_hash, hits in band.items():
                # If more than 1 document hashed to this bucket, it's a collision (candidate)
                if len(hits) > 1:
                    # 'hits' contains integers (self.counter). 
                    # Map those integers back to your actual JSON string IDs
                    actual_ids = [ids[hit] for hit in hits]
                    
                    candidates.add(tuple(actual_ids))   
        return candidates
    def get_candidates(self,signature):
        subvecs = self.make_subvecs(signature).astype(str)
        candidate_indices =set()

        for i,subvec in enumerate(subvecs):
            subvec_str = ','.join(subvec)
            if subvec_str in self.buckets[i]:
                candidate_indices.update(self.buckets[i][subvec_str])
        if not candidate_indices:
            return [] #No collisions 
        return candidate_indices
        
    

