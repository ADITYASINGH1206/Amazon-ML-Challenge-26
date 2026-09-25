import re
import unicodedata

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase and normalize unicode
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8').lower()
    return text

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = clean_text(name)
    
    # Standardize ampersand
    name = name.replace(' & ', ' and ')
    name = name.replace('&', ' and ')
    
    # Remove punctuation
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Map common legal endings
    legal_terms = {
        r'\bpvt\b': 'private',
        r'\bltd\b': 'limited',
        r'\binc\b': 'incorporated',
        r'\bllc\b': 'llc',
        r'\bcorp\b': 'corporation',
        r'\bco\b': 'company'
    }
    for k, v in legal_terms.items():
        name = re.sub(k, v, name)
        
    # Extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_address(address):
    if not isinstance(address, str):
        return ""
    address = clean_text(address)
    address = address.replace(' & ', ' and ')
    address = address.replace('&', ' and ')
    address = re.sub(r'[^\w\s]', ' ', address)
    
    address_terms = {
        r'\brd\b': 'road',
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bln\b': 'lane',
        r'\bdr\b': 'drive',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bste\b': 'suite',
        r'\bapt\b': 'apartment'
    }
    for k, v in address_terms.items():
        address = re.sub(k, v, address)
        
    address = re.sub(r'\s+', ' ', address).strip()
    return address

def extract_digits(text):
    if not isinstance(text, str):
        return ""
    return ' '.join(re.findall(r'\d+', text))

def tokenize_and_sort(text):
    if not text:
        return ""
    tokens = text.split()
    return ' '.join(sorted(tokens))
