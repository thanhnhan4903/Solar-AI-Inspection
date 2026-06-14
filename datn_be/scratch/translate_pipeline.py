import json
import re
import tokenize
from pathlib import Path
from io import BytesIO

def extract_strings_from_file(filepath):
    """Extracts all comments and docstrings from a Python file."""
    comments = set()
    docstrings = set()
    
    with open(filepath, 'rb') as f:
        content = f.read()
        
    try:
        tokens = list(tokenize.tokenize(BytesIO(content).readline))
    except Exception as e:
        print(f"Error tokenizing {filepath}: {e}")
        return comments, docstrings
        
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            # Strip the leading '#' and whitespace
            comment_text = tok.string.lstrip('#').strip()
            if comment_text:
                comments.add(comment_text)
        elif tok.type == tokenize.STRING:
            s = tok.string
            # Check if it is a docstring (starts with triple quotes)
            if s.startswith('"""') or s.startswith("'''"):
                # Strip the triple quotes
                doc_text = s[3:-3].strip()
                if doc_text:
                    docstrings.add(doc_text)
                    
    return comments, docstrings

def main_extract():
    all_comments = set()
    all_docstrings = set()
    
    # We will extract from:
    # 1. pv_fullsnap_universal_v34.py
    # 2. pv_fullsnap_universal_v42.py
    # 3. pv_fullsnap_universal_v55_full.py (outer part)
    
    files = [
        Path('scratch/pv_fullsnap_universal_v34.py'),
        Path('scratch/pv_fullsnap_universal_v42.py'),
        Path('scratch/pv_fullsnap_universal_v55_full.py')
    ]
    
    for filepath in files:
        if not filepath.exists():
            print(f"Skipping {filepath} (does not exist)")
            continue
        c, d = extract_strings_from_file(filepath)
        print(f"Extracted {len(c)} comments, {len(d)} docstrings from {filepath}")
        all_comments.update(c)
        all_docstrings.update(d)
        
    # We will save to scratch/translation_dictionary.json
    # Format: { "english": "" }
    # So the user or LLM can fill in the Vietnamese translations.
    out_dict = {}
    
    # Sort them to keep it clean
    for item in sorted(all_comments):
        out_dict[item] = ""
    for item in sorted(all_docstrings):
        out_dict[item] = ""
        
    out_path = Path('scratch/translation_dictionary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(out_dict)} unique strings to {out_path}")

if __name__ == '__main__':
    main_extract()
