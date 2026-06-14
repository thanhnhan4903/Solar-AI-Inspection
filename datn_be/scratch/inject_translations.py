import json
import tokenize
from io import BytesIO
from pathlib import Path

def translate_file_content(content, translation_dict):
    try:
        tokens = list(tokenize.tokenize(BytesIO(content.encode('utf-8')).readline))
    except Exception as e:
        print(f"Error tokenizing: {e}")
        return content

    # Split content into lines (keep line endings)
    lines = content.splitlines(keepends=True)

    # Filter comments and docstrings
    replaceable_tokens = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            comment_text = tok.string.lstrip('#').strip()
            if comment_text in translation_dict and translation_dict[comment_text]:
                replaceable_tokens.append(tok)
        elif tok.type == tokenize.STRING:
            s = tok.string
            if s.startswith('"""') or s.startswith("'''"):
                doc_text = s[3:-3].strip()
                if doc_text in translation_dict and translation_dict[doc_text]:
                    replaceable_tokens.append(tok)

    # Sort tokens by starting position in reverse order (bottom-up)
    # This prevents coordinate shifts of preceding lines/columns
    replaceable_tokens.sort(key=lambda t: t.start, reverse=True)

    for tok in replaceable_tokens:
        start_line, start_col = tok.start
        end_line, end_col = tok.end

        # Adjust to 0-based indices for lists
        start_idx = start_line - 1
        end_idx = end_line - 1

        if tok.type == tokenize.COMMENT:
            comment_text = tok.string.lstrip('#').strip()
            translated_val = translation_dict[comment_text]
            # Preserve the '#' characters and any leading whitespace within the comment
            prefix = tok.string[:len(tok.string) - len(tok.string.lstrip('#'))]
            spaces_count = len(tok.string.lstrip('#')) - len(comment_text)
            new_comment = prefix + (' ' * spaces_count) + translated_val
            
            # Replace comment in the line
            line_str = lines[start_idx]
            lines[start_idx] = line_str[:start_col] + new_comment + line_str[end_col:]

        elif tok.type == tokenize.STRING:
            doc_text = tok.string[3:-3].strip()
            translated_val = translation_dict[doc_text]
            q = tok.string[:3]
            
            # Reconstruct docstring
            # We want to keep original line endings of the file if possible (usually \n)
            new_doc = q + "\n" + translated_val + "\n" + q
            
            first_line_prefix = lines[start_idx][:start_col]
            # If end_idx is within bounds, get the suffix
            last_line_suffix = lines[end_idx][end_col:] if end_idx < len(lines) else ""
            
            # Replace range of lines [start_idx : end_idx + 1]
            new_block_str = first_line_prefix + new_doc + last_line_suffix
            new_block_lines = new_block_str.splitlines(keepends=True)
            
            lines[start_idx : end_idx + 1] = new_block_lines

    return "".join(lines)

def format_line_for_list(line):
    # Strip the trailing newline character
    if line.endswith('\r\n'):
        stripped = line[:-2]
    elif line.endswith('\n'):
        stripped = line[:-1]
    else:
        stripped = line
    # Escape single quotes and backslashes
    escaped = stripped.replace('\\', '\\\\').replace("'", "\\'")
    # Append the literal \n characters inside the python string literal
    return f"    '{escaped}\\n',\n"

def main():
    dict_path = Path('scratch/translation_dictionary.json')
    if not dict_path.exists():
        print("Error: Translation dictionary not found!")
        return

    with open(dict_path, 'r', encoding='utf-8') as f:
        translation_dict = json.load(f)

    # 1. Translate pv_fullsnap_universal_v34.py
    v34_path = Path('scratch/pv_fullsnap_universal_v34.py')
    print(f"Translating {v34_path}...")
    with open(v34_path, 'r', encoding='utf-8') as f:
        v34_content = f.read()
    translated_v34 = translate_file_content(v34_content, translation_dict)
    # Write back to translated file
    v34_translated_path = Path('scratch/pv_fullsnap_universal_v34_translated.py')
    with open(v34_translated_path, 'w', encoding='utf-8') as f:
        f.write(translated_v34)

    # 2. Translate pv_fullsnap_universal_v42.py
    v42_path = Path('scratch/pv_fullsnap_universal_v42.py')
    print(f"Translating {v42_path}...")
    with open(v42_path, 'r', encoding='utf-8') as f:
        v42_content = f.read()
    translated_v42 = translate_file_content(v42_content, translation_dict)
    v42_translated_path = Path('scratch/pv_fullsnap_universal_v42_translated.py')
    with open(v42_translated_path, 'w', encoding='utf-8') as f:
        f.write(translated_v42)

    # 3. Read the original pv_fullsnap_universal_v55_full.py to get header/footer and rebuild
    v55_path = Path('scratch/pv_fullsnap_universal_v55_full.py')
    print(f"Reading {v55_path} structure...")
    with open(v55_path, 'r', encoding='utf-8') as f:
        v55_lines = f.readlines()

    # Find the bounds of _ENGINE_V34_LINES and _ENGINE_V42_LINES
    v34_start_idx = -1
    v34_end_idx = -1
    v42_start_idx = -1
    v42_end_idx = -1

    for idx, line in enumerate(v55_lines):
        if line.startswith('_ENGINE_V34_LINES = ['):
            v34_start_idx = idx
        elif v34_start_idx != -1 and v34_end_idx == -1 and line.startswith(']'):
            v34_end_idx = idx
        elif line.startswith('_ENGINE_V42_LINES = ['):
            v42_start_idx = idx
        elif v42_start_idx != -1 and v42_end_idx == -1 and line.startswith(']'):
            v42_end_idx = idx

    if -1 in (v34_start_idx, v34_end_idx, v42_start_idx, v42_end_idx):
        print("Error: Could not locate engine array boundaries in v55 file!")
        return

    # Extract original header (lines before _ENGINE_V34_LINES)
    header_content = "".join(v55_lines[:v34_start_idx])
    # Extract footer (lines after _ENGINE_V42_LINES)
    footer_content = "".join(v55_lines[v42_end_idx + 1:])

    # Translate header and footer
    print("Translating header and footer...")
    translated_header = translate_file_content(header_content, translation_dict)
    translated_footer = translate_file_content(footer_content, translation_dict)

    # Now, format the translated v34 and v42 files as list-of-strings
    print("Formatting embedded engines...")
    with open(v34_translated_path, 'r', encoding='utf-8') as f:
        v34_trans_lines = f.readlines()
    v34_embedded_lines = [format_line_for_list(l) for l in v34_trans_lines]

    with open(v42_translated_path, 'r', encoding='utf-8') as f:
        v42_trans_lines = f.readlines()
    v42_embedded_lines = [format_line_for_list(l) for l in v42_trans_lines]

    # Re-assemble the final v55 file contents
    final_v55_lines = []
    final_v55_lines.append(translated_header)
    final_v55_lines.append("_ENGINE_V34_LINES = [\n")
    final_v55_lines.extend(v34_embedded_lines)
    final_v55_lines.append("]\n")
    # Add spacing between arrays if needed, let's keep it clean
    final_v55_lines.append("\n")
    final_v55_lines.append("_ENGINE_V42_LINES = [\n")
    final_v55_lines.extend(v42_embedded_lines)
    final_v55_lines.append("]\n")
    final_v55_lines.append(translated_footer)

    output_path = Path('scratch/pv_fullsnap_universal_v55_full.py')
    # Backup original first
    backup_path = Path('scratch/pv_fullsnap_universal_v55_full.py.bak')
    if not backup_path.exists():
        shutil = __import__('shutil')
        shutil.copy2(output_path, backup_path)
        print(f"Backed up original file to {backup_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("".join(final_v55_lines))

    print(f"Successfully generated translated {output_path}!")

if __name__ == '__main__':
    main()
