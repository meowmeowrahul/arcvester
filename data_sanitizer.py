import re
def clean_json(text):
    """Takes Text as input and returns Text with \n \t and trailing spaces removed"""
    if not text:
        return ""
    
    #Removing \n and \t
    text = text.replace("\n",' ').replace("\t",' ')
    
    #Removing multiple spaces as one
    text = re.sub(r'\s+',' ',text)
    
    return text.strip()
def sanitize_arxiv_record(record: dict)-> dict:
    record_id = record.get("id",'')
    title = record.get('title','')
    abstract = record.get('abstract','')
    categories= record.get('categories','')
    
    clean_title=clean_json(title)
    clean_abstract=clean_json(abstract)
    
    
    return {
        "id":record_id,
        "categories":categories,
        "clean_title":clean_title,
        "clean_abstract":clean_abstract
    }
    