import json
from data_sanitizer import sanitize_arxiv_record
from tokenizer import tokenizer
from inverted_index import InvertedIndex

def run_sanitization_stage(input_path,output_path):
    processed_count = 0
    print("Starting data sanitization...")
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            try:
                record=json.loads(line)
            except json.JSONDecodeError:
                continue   #Skip broken lines   
            
            clean_record=sanitize_arxiv_record(record)
            outfile.write(json.dumps(clean_record)+ '\n')
            processed_count +=1
            if processed_count % 50 == 0:
                print(f"  ...sanitized {processed_count} documents.")
    print(f"Success! Processed and cleaned {processed_count} papers.")
    
def run_tokenization_stage(input_path,output_path):
    processed_count = 0
    print("Starting Tokenization...")
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            try:
                record=json.loads(line)
            except json.JSONDecodeError:
                continue   #Skip broken lines   
            
            clean_title=tokenizer(record.get("clean_title",''))
            clean_abstract=tokenizer(record.get("clean_abstract",''))
            clean_record={
                "id":record.get('id',''),
                "categories":record.get('categories',''),
                "clean_title":clean_title,
                "clean_abstract":clean_abstract
            }
            outfile.write(json.dumps(clean_record)+ '\n')
            
            processed_count +=1
            if processed_count % 50 == 0:
                print(f"  ...tokenized {processed_count} documents.")
    print(f"Success! Output File Path:-{output_path} ")
def run_indexation_stage(input_path,output_path):
    print("Starting Indexing.....")
    inverted = InvertedIndex()
    
    processed_count = 0
    
    with open(input_path,'r') as infile:
        for line in infile:
            try:
                record=json.loads(line)
            except json.JSONDecodeError:
                continue
            
            doc_id = record.get("id")
            title_tokens = record.get('clean_title')
            abstract_tokens= record.get('clean_abstract')
            
            if doc_id:
                inverted.add_document(doc_id,"title",title_tokens)
                inverted.add_document(doc_id,"abstract",abstract_tokens)
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  ...indexed {processed_count} documents.")
    inverted.save_to_disk(output_path) 
    print(f"Successfully Indexed:{processed_count} documents")       
    
if __name__ == "__main__":
    raw_data_file = "./archive/metadata-200.json"
    sanitized_file = "./archive/output-200.json"
    tokenized_file = './archive/output-200-tokenized.json'
    indexed_file = './archive/output-200-indexed.json'
    run_sanitization_stage(raw_data_file,sanitized_file)
    run_tokenization_stage(sanitized_file,tokenized_file)
    run_indexation_stage(tokenized_file,indexed_file)
