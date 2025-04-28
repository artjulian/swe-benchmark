import jsonlines
import os
import subprocess
import tempfile
import shutil

DATASET_DIR = "data/swe-bench/"
INSTANCE_FILE = os.path.join(DATASET_DIR, "swe_bench_data.jsonl")
REPOS_DIR = "cloned_repos"

def load_instance(instance_file, index=0):
    with jsonlines.open(instance_file) as reader:
        for i, item in enumerate(reader):
            if i == index:
                return item
    raise IndexError("Instance not found")

def count_instances(instance_file):
    count = 0
    with jsonlines.open(instance_file) as reader:
        for _ in reader:
            count += 1
    return count

def clone_repo(instance, repos_dir):
    repo_url = f"https://github.com/{instance['repo']}.git"
    base_commit = instance["base_commit"]
    repo_name = instance['repo'].replace('/', '_')
    repo_dir = os.path.join(repos_dir, repo_name)

    if os.path.exists(repo_dir):
        print(f"[ℹ] Repository {repo_name} already exists, skipping...")
        return repo_dir

    print(f"[ℹ] Cloning {repo_name} at commit {base_commit}")
    subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    subprocess.run(["git", "checkout", base_commit], cwd=repo_dir, check=True)
    return repo_dir

def main():
    # Create repos directory if it doesn't exist
    os.makedirs(REPOS_DIR, exist_ok=True)
    
    # Count total instances first
    total_instances = count_instances(INSTANCE_FILE)
    print(f"\nFound {total_instances} instances to process")

    # Clone all repositories
    for index in range(total_instances):
        try:
            instance = load_instance(INSTANCE_FILE, index=index)
            instance_id = instance["instance_id"]
            
            print(f"\nProcessing instance {index + 1}/{total_instances}: {instance_id}")
            clone_repo(instance, REPOS_DIR)
            
        except Exception as e:
            print(f"[✗] Failed to process instance {index + 1}: {e}")
            continue

    print("\nAll repositories have been cloned successfully!")

if __name__ == "__main__":
    main() 