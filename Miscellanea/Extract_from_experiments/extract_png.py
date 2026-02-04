import os
import shutil

def extract_png_images():
    # 1. Setup: Root directory and Destination configuration
    root_dir = os.getcwd()
    output_folder_name = "ceg"
    dest_dir = os.path.join(root_dir, output_folder_name)

    # 2. Preparation: Create the destination folder
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination folder: {dest_dir}")
    else:
        print(f"Destination folder '{output_folder_name}' already exists.")

    print("-" * 30)
    count = 0

    # 3. Execution: Recursively walk through directories
    for current_root, dirs, files in os.walk(root_dir):
        
        # Prevent recursion: Do not look inside the destination folder
        if output_folder_name in dirs:
            dirs.remove(output_folder_name)

        for filename in files:
            # A. Filter: Look strictly for .png files (case insensitive)
            if not filename.lower().endswith(".png"):
                continue

            # 4. Copy Logic
            source_file = os.path.join(current_root, filename)
            destination_file = os.path.join(dest_dir, filename)

            # --- DUPLICATE HANDLING ---
            # If a file with the same name exists, append the parent folder name
            if os.path.exists(destination_file):
                parent_folder = os.path.basename(current_root)
                name, ext = os.path.splitext(filename)
                
                # New name format: image_ParentFolder.png
                new_name = f"{name}_{parent_folder}{ext}"
                destination_file = os.path.join(dest_dir, new_name)

            try:
                shutil.copy2(source_file, destination_file)
                print(f"[Copied] {filename}")
                count += 1
            except Exception as e:
                print(f"[Error] Could not copy {filename}: {e}")

    print("-" * 30)
    print(f"Process finished. Total PNG files extracted: {count}")

if __name__ == "__main__":
    extract_png_images()