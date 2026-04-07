import os
import emoji

#Configuration
PROJECT_ROOT = '.'  # Current directory
DRY_RUN = False      # Set to False to actually delete data
SKIP_DIRS = {'.git', '.idea', 'node_modules', '__pycache__', 'venv', 'env'}
SKIP_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.exe', '.pyc', '.zip', '.pdf'}

def remove_emojis(text):
    """
    Replaces all emoji characters with an empty string.
    """
    return emoji.replace_emoji(text, replace='')

def process_files():
    print(f"--- STARTING PROCESS (Dry Run: {DRY_RUN}) ---")
    
    files_changed = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Modify dirs in-place to skip specific directories during traversal
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)

            if ext.lower() in SKIP_EXTENSIONS:
                continue

            try:
                # Attempt to read file as UTF-8
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # check for emojis
                if emoji.emoji_count(content) > 0:
                    clean_content = remove_emojis(content)
                    
                    # Double check content actually changed
                    if content != clean_content:
                        print(f"[{'DRY RUN' if DRY_RUN else 'CLEANING'}] Found emojis in: {file_path}")
                        
                        if not DRY_RUN:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(clean_content)
                        
                        files_changed += 1

            except UnicodeDecodeError:
                # Skip binary files that aren't UTF-8 text
                continue
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"--- COMPLETED. Files impacted: {files_changed} ---")

if __name__ == "__main__":
    process_files()