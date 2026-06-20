import numpy as np 
from itertools import combinations
class LSH:
    def __init__(self,b):
        self.buckets = []
        self.counter = 0
        self.b = b
        for i in range(b):
            self.buckets.append({})
        
    def make_subvecs(self,signature):
        l = len(signature)
        assert l%self.b==0
        r = int(l/self.b)
        subvec = []
        for i in range(0,l,r):
            subvec.append(signature[i:i+r])
        return np.stack(subvec)
    def add_hash(self,signature):
        subvec = self.make_subvecs(signature).astype(str)
        for i,subvec in enumerate(subvec):
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
                    
                    # Generate all pairs from the documents in this bucket
                    for pair in combinations(actual_ids, 2):
                        # Sort the tuple so ("A", "B") and ("B", "A") are recognized as the same pair
                        candidates.add(tuple(sorted(pair)))
                        
        return candidates


