import os
import json
import subprocess
import tempfile
import shutil
import requests
import jsonlines
from pathlib import Path

# === CONFIGURATION === #
API_URL = "https://api-acc.polychat.com/v1/chat/completions"
MODEL = "polychat/3x/sonnet/sonnet_and_gpt4o"
USER_ID = "1"
PROMPT_TYPE = "code"
INSTANCE_FILE = "data/swe-bench/swe_bench_verified.jsonl"
RESULT_DIR = "results"
DOCKER_TAG_NAMESPACE = "local"

os.makedirs(RESULT_DIR, exist_ok=True)

def load_instance(instance_id):
    with jsonlines.open(INSTANCE_FILE) as reader:
        for item in reader:
            if item["instance_id"] == instance_id:
                return item
    raise ValueError(f"Instance ID {instance_id} not found.")

def clone_repo(instance):
    repo_url = f"https://github.com/{instance['repo']}.git"
    base_commit = instance['base_commit']

    temp_dir = tempfile.mkdtemp(prefix="swebench_repo_")
    subprocess.run(["git", "clone", repo_url, temp_dir], check=True)
    subprocess.run(["git", "checkout", base_commit], cwd=temp_dir, check=True)
    return temp_dir

def generate_patch(prompt_text):
    payload = {
        "model": MODEL,
        "stream": False,
        "user": USER_ID,
        "prompt_type": PROMPT_TYPE,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
    }
    response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def write_patch_file(patch_text, directory):
    patch_path = os.path.join(directory, "patch.diff")
    with open(patch_path, "w") as f:
        f.write(patch_text)
    return patch_path

def apply_patch(repo_dir, patch_path):
    subprocess.run(["git", "apply", patch_path], cwd=repo_dir, check=True)

def generate_dockerfile(repo_dir, test_file, test_case):
    dockerfile = f"""
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install . pytest
CMD ["pytest", "{test_file}::{test_case}"]
"""
    with open(os.path.join(repo_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile)

def build_docker_image(tag, context_dir):
    subprocess.run(["docker", "build", "-t", tag, context_dir], check=True)

def run_docker_container(tag):
    result = subprocess.run(["docker", "run", "--rm", tag], capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def cleanup_docker(tag):
    subprocess.run(["docker", "rmi", "-f", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def build_and_test_instance(instance_id):
    instance = load_instance(instance_id)
    tag = f"{DOCKER_TAG_NAMESPACE}/sweb.eval.arm64.{instance_id}"

    repo_dir = clone_repo(instance)
    try:
        # Generate patch
        prompt = f"""Generate a valid git patch to fix this issue:
Repository: {instance['repo']}
Problem: {instance['problem_statement']}
Hints: {instance.get('hints_text', '')}
"""
        patch_text = generate_patch(prompt)
        patch_path = write_patch_file(patch_text, repo_dir)
        try:
          apply_patch(repo_dir, patch_path)
        except subprocess.CalledProcessError as e:
          print("Patch apply failed. Patch preview:")
          print(open(patch_path).read())
          raise

        # Dockerize
        generate_dockerfile(repo_dir, instance['test_file'], instance['test_case'])
        build_docker_image(tag, repo_dir)

        # Run container
        stdout, stderr, code = run_docker_container(tag)
        result = {
            "instance_id": instance_id,
            "exit_code": code,
            "stdout": stdout,
            "stderr": stderr
        }

        with open(os.path.join(RESULT_DIR, f"result_{instance_id}.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"[✓] Completed: {instance_id} (exit={code})")

    except Exception as e:
        print(f"[✗] Failed: {instance_id}: {e}")

    finally:
        cleanup_docker(tag)
        shutil.rmtree(repo_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python build_and_test_instance.py <instance_id>")
        exit(1)
    build_and_test_instance(sys.argv[1])
