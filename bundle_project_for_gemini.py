import os
import fnmatch

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "project_context_for_gemini.md")

# Directories to ignore
IGNORED_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".git",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "weights",
    "data",
    "scratch",
    "db",
    "__tmp__",
}

# File names or patterns to ignore
IGNORED_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "project_context_for_gemini.md",
    "bundle_project_for_gemini.py",
    "Dataset.zip",
    "TEST.zip",
    "epc solar.png",
    "solar farm.png",
}

# File extensions to ignore (mostly binary or non-text files)
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", 
    ".zip", ".tar", ".gz", ".7z", ".rar", 
    ".pt", ".pth", ".onnx", ".h5", ".bin", 
    ".exe", ".dll", ".so", ".dylib", 
    ".db", ".sqlite", ".sqlite3",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".pyd"
}

# Mapping extensions to markdown language specifiers
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".md": "markdown",
    ".sh": "bash",
    ".bat": "cmd",
    ".ps1": "powershell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".ini": "ini",
    ".env": "env",
    "Dockerfile": "dockerfile"
}

def is_ignored(path, is_dir=False):
    name = os.path.basename(path)
    if is_dir:
        return name in IGNORED_DIRS
    
    # Check ignored names
    if name in IGNORED_FILES:
        return True
    
    # Check ignored extensions
    _, ext = os.path.splitext(name)
    if ext.lower() in IGNORED_EXTENSIONS:
        return True
    
    # Check for temporary/hidden files
    if name.startswith("."):
        # Let some dotfiles like .env, .gitignore pass
        if name not in {".gitignore", ".env", ".env.example"}:
            return True
            
    return False

def generate_tree(dir_path, prefix=""):
    """Generates a tree structure of the project as a string."""
    tree_str = ""
    try:
        entries = sorted(os.scandir(dir_path), key=lambda e: (not e.is_dir(), e.name.lower()))
    except Exception as e:
        return f"[Error listing directory {dir_path}: {e}]\n"
        
    entries = [e for e in entries if not is_ignored(e.path, e.is_dir())]
    
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        
        tree_str += f"{prefix}{connector}{entry.name}\n"
        
        if entry.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree_str += generate_tree(entry.path, new_prefix)
            
    return tree_str

def gather_files(dir_path):
    """Walks the directory and returns list of relative paths for files to read."""
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), is_dir=True)]
        
        for file in files:
            full_path = os.path.join(root, file)
            if not is_ignored(full_path, is_dir=False):
                rel_path = os.path.relpath(full_path, dir_path)
                file_list.append((rel_path, full_path))
    return sorted(file_list, key=lambda x: x[0].lower())

def bundle_project():
    print(f"Scanning directory: {PROJECT_ROOT}")
    
    # 1. Generate directory tree
    print("Generating project directory tree...")
    tree = generate_tree(PROJECT_ROOT)
    
    # 2. Gather all code files
    print("Gathering files to bundle...")
    files = gather_files(PROJECT_ROOT)
    
    total_files = len(files)
    print(f"Found {total_files} files to bundle.")
    
    # 3. Write output file
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Header
        f.write("# PROJECT CONTEXT: SOLAR AI INSPECTION\n\n")
        f.write("This file contains the complete repository structure and code files for the Solar AI Inspection project.\n")
        f.write("It includes both Backend (`datn_be`) and Frontend (`datn_fe`) components.\n\n")
        
        # Directory Structure Section
        f.write("## Project Directory Structure\n")
        f.write("```text\n")
        f.write(tree)
        f.write("```\n\n")
        
        f.write("## Source Code Files\n\n")
        
        # Individual File Contents
        for idx, (rel_path, full_path) in enumerate(files, 1):
            normalized_path = rel_path.replace("\\", "/")
            print(f"[{idx}/{total_files}] Processing: {normalized_path}")
            
            f.write(f"### File: `{normalized_path}`\n\n")
            
            # Determine language syntax highlighting
            ext = os.path.splitext(rel_path)[1]
            lang = LANGUAGE_MAP.get(ext, LANGUAGE_MAP.get(os.path.basename(rel_path), ""))
            
            # Read contents
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as code_file:
                    content = code_file.read()
                
                # Check for triple backticks in content and replace them to not break markdown
                safe_content = content.replace("```", "'''")
                
                f.write(f"```{lang}\n")
                f.write(safe_content)
                if not safe_content.endswith("\n"):
                    f.write("\n")
                f.write("```\n\n")
                f.write("---\n\n")
            except Exception as e:
                f.write(f"*[Error reading file: {e}]*\n\n---\n\n")
                print(f"Error reading {normalized_path}: {e}")
                
    print("\nBundling completed successfully!")
    print(f"Output file: {OUTPUT_FILE}")
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")
    print("\nYou can now upload this single markdown file directly into Gemini Advanced or Google AI Studio!")

if __name__ == "__main__":
    bundle_project()
