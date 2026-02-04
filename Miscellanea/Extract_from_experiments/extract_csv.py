import os
import shutil

def organize_csv_files_by_category():
    # 1. Setup: Root directory where the script runs
    root_dir = os.getcwd()

    # 2. Define common criteria (suffixes apply to all categories)
    valid_suffixes = [
        "_ALL_agregate_then_metric.csv",
        "_ALL_metric_then_aggregate.csv",
        "_metric_then_aggregate_flat.csv",
        "_metric_then_aggregate_report.csv"
    ]

    # 3. Configuration: Define categories, target folders, and prefixes
    # We use a list of dictionaries to easily map prefixes to specific folders
    categories = [
        {
            "folder": "csv_performance",
            "prefix": "metrics_performance_by_range_"
        },
        {
            "folder": "csv_number_of_points",
            "prefix": "number_of_points_by_zones_"
        },
        {
            "folder": "csv_out_of_limits",
            "prefix": "predictions_out_of_limits_"
        }
    ]

    # 4. Preparation: Create destination folders and prepare exclusion list
    destination_folder_names = []

    for cat in categories:
        dest_path = os.path.join(root_dir, cat["folder"])

        # Create folder if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
            print(f"Folder created: {cat['folder']}")
        else:
            print(f"Folder '{cat['folder']}' already exists.")

        # Add to list to prevent the script from scanning these folders later
        destination_folder_names.append(cat["folder"])

    print("-" * 30)
    total_copied_count = 0

    # 5. Execution: Recursively walk through directories
    for current_root, dirs, files in os.walk(root_dir):

        # Prevent recursion: Do not look inside the destination folders we just created
        # Modifying 'dirs' in-place affects os.walk behavior
        for d in destination_folder_names:
            if d in dirs:
                dirs.remove(d)

        for filename in files:
            # A. Filter: Check if file ends with a valid suffix
            if not filename.endswith(tuple(valid_suffixes)):
                continue

            # B. Match: Determine which category the file belongs to based on prefix
            matched_category = None
            for cat in categories:
                if filename.startswith(cat["prefix"]):
                    matched_category = cat
                    break # Stop checking other categories once found

            # If no prefix matches, skip this file
            if not matched_category:
                continue

            # 6. Copy Logic
            dest_dir = os.path.join(root_dir, matched_category["folder"])
            source_file = os.path.join(current_root, filename)
            destination_file = os.path.join(dest_dir, filename)

            # --- DUPLICATE HANDLING ---
            # If a file with the same name exists (from a different subfolder),
            # append the parent folder name to preserve both files.
            if os.path.exists(destination_file):
                parent_folder = os.path.basename(current_root)
                name, ext = os.path.splitext(filename)
                # Example: file_ParentFolderName.csv
                new_name = f"{name}_{parent_folder}{ext}"
                destination_file = os.path.join(dest_dir, new_name)

            try:
                shutil.copy2(source_file, destination_file)
                print(f"[Copied] {filename} -> {matched_category['folder']}/")
                total_copied_count += 1
            except Exception as e:
                print(f"[Error] Could not copy {filename}: {e}")

    print("-" * 30)
    print(f"Process finished. Total files copied: {total_copied_count}")

if __name__ == "__main__":
    organize_csv_files_by_category()