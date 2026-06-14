import tokenize
from io import BytesIO

def extract_comments_and_docstrings(filepath):
    comments = []
    docstrings = []
    with open(filepath, 'rb') as f:
        try:
            tokens = list(tokenize.tokenize(f.readline))
        except Exception as e:
            print(f'Error tokenizing {filepath}: {e}')
            return [], []
            
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                comments.append((tok.start[0], tok.string))
            elif tok.type == tokenize.STRING:
                s = tok.string
                if s.startswith('"""') or s.startswith("'''"):
                    docstrings.append((tok.start[0], s))
    return comments, docstrings

for path in ['scratch/pv_fullsnap_universal_v34.py', 'scratch/pv_fullsnap_universal_v42.py']:
    c, d = extract_comments_and_docstrings(path)
    print(f'{path}: {len(c)} comments, {len(d)} docstrings')
