from collections import defaultdict
import json
class InvertedIndex:
    def __init__(self):
        self.index=defaultdict(list)
        self.document_lengths=defaultdict(dict)
    def add_document(self,doc_id,field_name,tokens):
        self.document_lengths[doc_id][field_name]=len(tokens) # For BM25 ranking shit
        
        term_counts=defaultdict(int)
        
        for token in tokens:
            term_counts[token] +=1
        
        for term, count in term_counts.items():
            # Check if this document is already in the postings list for this term
            posting_exists = False
            for i, (existing_doc_id, field_counts) in enumerate(self.index[term]):
                if existing_doc_id == doc_id:
                    # Document already exists, just update the field count
                    self.index[term][i][1][field_name] = count
                    posting_exists = True
                    break
            
            if not posting_exists:
                # This is the first time we see this term for this document
                self.index[term].append([doc_id, {field_name: count}])
    def save_to_disk(self,path:str):
        data_to_save={
            "index":self.index,
            "document_lengths":self.document_lengths
        }     
        with open(path,'w') as f:
            json.dump(data_to_save,f)
        print(f"Index saved to {path}")
    def load_from_disk(self,path:str):
        with open(path, 'r') as f:
            data = json.load(f)
            # defaultdict requires a factory, so we load and convert
            self.index = defaultdict(list, data['index'])
            self.document_lengths = defaultdict(dict,data['doc_lengths'])
        print(f"Index loaded from {path}")
