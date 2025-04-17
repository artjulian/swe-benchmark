import requests
import json
import os
import jsonlines
import subprocess
import tempfile
import shutil

API_URL = "https://api-acc.polychat.com/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json"
}

MODEL = "polychat/3x/sonnet/sonnet_and_gpt4o"
PROMPT_TYPE = "code"
USER_ID = "1"

PATCH_DIR = "my_patches"
RESULTS_DIR = "results"
DATASET_DIR = "data/swe-bench/"
INSTANCE_FILE = os.path.join(DATASET_DIR, "swe_bench_verified.jsonl")

os.environ["SWEBENCH_FORCE_PLATFORM"] = "arm64"
os.makedirs(PATCH_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_instance(instance_file, index=0):
    with jsonlines.open(instance_file) as reader:
        for i, item in enumerate(reader):
            if i == index:
                return item
    raise IndexError("Instance not found")

def generate_code_from_api(prompt_text):
    payload = {
        "model": MODEL,
        "stream": False,
        "user": USER_ID,
        "prompt_type": PROMPT_TYPE,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ]
    }

    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
    response.raise_for_status()
    result = response.json()

    # Change this based on how your API structures its response
    return result["choices"][0]["message"]["content"]

def save_patch(instance_id, code):
    patch_path = os.path.join(PATCH_DIR, f"{instance_id}.patch")
    with open(patch_path, "w") as f:
        f.write(code)
    return patch_path

def clone_repo(instance):
    repo_url = f"https://github.com/{instance['repo']}.git"
    base_commit = instance["base_commit"]

    temp_dir = tempfile.mkdtemp(prefix="swebench_repo_")
    subprocess.run(["git", "clone", repo_url, temp_dir], check=True)
    subprocess.run(["git", "checkout", base_commit], cwd=temp_dir, check=True)
    return temp_dir

def apply_patch(repo_dir, patch_path):
    subprocess.run(["git", "apply", patch_path], cwd=repo_dir, check=True)

def generate_dockerfile(repo_dir, test_file, test_case):
    dockerfile = f"""\
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

def create_predictions_file(instance_id, patch_path):
    predictions_path = os.path.join(PATCH_DIR, f"predictions_{instance_id}.json")
    predictions = {
        instance_id: {
            "instance_id": instance_id,
            "model_name_or_path": MODEL,
            "model_patch": open(patch_path, "r").read()
        }
    }
    with open(predictions_path, "w") as f:
        json.dump(predictions, f)
    return predictions_path

def main():
    instance = load_instance(INSTANCE_FILE, index=0)  # change index for batch
    instance_id = instance["instance_id"]
    prompt = f"""Please generate a code patch for the following issue:\n
Problem: {instance['problem_statement']}\n
Hints: {instance.get('hints_text', '')}\n
Repository: {instance['repo']}\n
Return only valid git patch format.
"""
    code = generate_code_from_api(prompt)
    patch_path = save_patch(instance_id, code)
    print(f"Generated patch saved to {patch_path}")

    tag = f"local/sweb.eval.arm64.{instance_id}"
    repo_dir = clone_repo(instance)

    try:
        try:
          apply_patch(repo_dir, patch_path)
        except subprocess.CalledProcessError as e:
          print("Patch apply failed. Patch preview:")
          print(open(patch_path).read())
          raise

        generate_dockerfile(repo_dir, instance["test_file"], instance["test_case"])
        build_docker_image(tag, repo_dir)
        stdout, stderr, code = run_docker_container(tag)

        result_path = os.path.join(RESULTS_DIR, f"result_{instance_id}.json")
        with open(result_path, "w") as f:
            json.dump({
                "instance_id": instance_id,
                "exit_code": code,
                "stdout": stdout,
                "stderr": stderr
            }, f, indent=2)

        print(f"[✓] Test complete: {instance_id} (exit={code})")
    except Exception as e:
        print(f"[✗] Failed: {instance_id} – {e}")
    finally:
        cleanup_docker(tag)
        shutil.rmtree(repo_dir)

if __name__ == "__main__":
    main()