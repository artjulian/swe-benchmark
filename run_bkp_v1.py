import requests
import json
from unidiff import PatchSet
import os
import jsonlines
import subprocess
import tempfile
import shutil

API_URL = "http://localhost:8000/v1/code/completions"
HEADERS = {
    "Content-Type": "application/json"
}

MODEL = "polychat/2x/sonnet/sonnet_and_gpt4o"
PROMPT_TYPE = "code"
USER_ID = "1"
ONLY_CODE = True

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

def generate_code_from_api(prompt_text, target_file, file_content):
    def number_lines(text):
      return "\n".join(f"{i+1}: {line}" for i, line in enumerate(text.splitlines()))

    numbered_content = number_lines(file_content)

    prompt = f"""Please generate a code patch for the following issue:

Problem: {prompt_text}

The issue is in the following file: `{target_file}`

Generate a patch in valid git-patch format that fixes the issue.

Requirements:
1. Use exactly 4 spaces for indentation
2. Show correct number of lines in patch header
3. Only use functions defined in the codebase
4. Handle all operators in _operators dictionary
5. Remove any trailing whitespace in the patch
6. Match the exact line numbers in the file content, the file in file_content starts at line 1
7. Do not include any other text or comments in the patch
8. Match the exact indentation and spacing used in the file content
9. Do not add leading or trailing blank lines before or after the code you modify.
10. Preserve existing whitespace from the file exactly.
11. Do not include the `index` line in the diff header.

Return only the git patch format, nothing else.

Current file content:
<file_content>
{numbered_content}
</file_content>"""

    payload = {
        "model": MODEL,
        "stream": False,
        "user": USER_ID,
        "prompt_type": PROMPT_TYPE,
        "only_code": ONLY_CODE,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    print(f"[DEBUG] Payload: {payload}")

    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
    response.raise_for_status()
    result = response.json()

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

def apply_unidiff_patch(patch_path, repo_dir):
    with open(patch_path, 'r') as f:
        patch = PatchSet(f)

    for patched_file in patch:
        file_path = os.path.join(repo_dir, patched_file.path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for hunk in patched_file:
            start = hunk.source_start - 1
            end = start + hunk.source_length
            expected = [line.value for line in hunk.source_lines]
            if lines[start:end] != expected:
                raise ValueError(f"Mismatch in file {file_path} at hunk starting on line {start+1}")
            lines[start:end] = [line.value for line in hunk.target_lines]

        with open(file_path, 'w') as f:
            f.writelines(lines)

    print(f"✅ Patch applied successfully to {len(patch)} file(s).")

def apply_unidiff_patch_relaxed(patch_path, repo_dir):
    try:
        with open(patch_path, 'r') as f:
            patch = PatchSet(f)
    except UnidiffParseError as e:
        raise RuntimeError(f"Failed to parse patch: {e}")

    for patched_file in patch:
        file_path = os.path.join(repo_dir, patched_file.path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for hunk in patched_file:
            start = hunk.source_start - 1
            expected = [l.value for l in hunk.source_lines]
            actual = lines[start:start + len(expected)]

            if expected != actual:
                print("⚠️ Warning: source lines did not match exactly, continuing anyway.")

            lines[start:start + len(expected)] = [l.value for l in hunk.target_lines]

        with open(file_path, 'w') as f:
            f.writelines(lines)

    print(f"✅ Patch applied with relaxed validation to {len(patch)} file(s).")

def apply_patch(repo_dir, patch_path):
    abs_patch_path = os.path.abspath(patch_path)
    
    # Print the target file content before patching
    target_file = None
    with open(patch_path, 'r') as f:
        for line in f:
            if line.startswith('--- a/'):
                target_file = os.path.join(repo_dir, line[6:].strip())
                break
    
    # if target_file and os.path.exists(target_file):
    #     print("Target file content:")
    #     with open(target_file, 'r') as f:
    #         print(f.read())
    
    validate_patch_format(abs_patch_path)

    # Try to apply patch with more verbosity
    try:
        # First try with --verbose to see what git is trying to do
        result = subprocess.run(
            ["git", "apply", "--verbose", "--check", abs_patch_path],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        print("Verbose patch check output:")
        print(result.stdout)
        print(result.stderr)
        
        # If verbose check passes, try actual application
        subprocess.run(["git", "apply", abs_patch_path], cwd=repo_dir, check=True)
    except subprocess.CalledProcessError as e:
        print("Patch validation failed. Patch preview:")
        print(open(patch_path).read())
        print("\nPatch check output:")
        print(e.stderr)
        raise

def apply_full_file_replacement(repo_dir, patch_text):
    import re

    # Extract file path
    file_path = None
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) == 2:
                file_path = parts[1].strip()
                break
    if not file_path:
        raise ValueError("Could not extract target file path from patch")

    full_path = os.path.join(repo_dir, file_path)
    print(f"[DEBUG] Replacing file: {full_path}")

    # Extract new file content from patch
    lines = patch_text.splitlines()
    new_content = []
    inside_hunk = False
    for line in lines:
        if line.startswith("@@"):
            inside_hunk = True
            continue
        if inside_hunk:
            if line.startswith("+") and not line.startswith("+++ "):
                new_content.append(line[1:] + "\n")
            elif not line.startswith("-") and not line.startswith("--- "):
                new_content.append(line + "\n")

    # Write new content
    with open(full_path, "w") as f:
        f.writelines(new_content)

    print(f"✅ Full file replacement done for {file_path}")

def generate_dockerfile(repo_dir, test_file, test_case):
    dockerfile = f"""\
FROM python:3.10-slim
WORKDIR /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    make \\
    pkg-config \\
    gfortran \\
    python3-dev

# Install Python build dependencies with specific versions
RUN pip install --no-cache-dir \\
    "setuptools==57.5.0" \\
    "wheel>=0.37.0" \\
    "numpy>=1.20" \\
    "cython==0.29.22" \\
    "extension-helpers>=1.0"

# Install astropy in development mode
RUN pip install --no-cache-dir -v -e .

# Install test dependencies
RUN pip install --no-cache-dir pytest

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
            "model_patch": open(patch_path, "r").read(),
            "model_name_or_path": MODEL,
            "completion_tokens": None,
            "total_tokens": None
        }
    }
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)
    return predictions_path

def extract_file_context(repo_dir, file_path, lines_before=20, lines_after=20):
    full_path = os.path.join(repo_dir, file_path)
    if not os.path.exists(full_path):
        return None
    with open(full_path, "r") as f:
        lines = f.readlines()
    # Optional: return whole file or nearby lines
    return "".join(lines)
    
def validate_patch_format(patch_path):
    with open(patch_path, 'r') as f:
        lines = f.readlines()
        
    # Basic patch format validation
    if not any(line.startswith('diff --git') for line in lines):
        raise ValueError("Patch missing diff --git line")
    if not any(line.startswith('--- a/') for line in lines):
        raise ValueError("Patch missing --- a/ line")
    if not any(line.startswith('+++ b/') for line in lines):
        raise ValueError("Patch missing +++ b/ line")
    if not any(line.startswith('@@ ') for line in lines):
        raise ValueError("Patch missing @@ line")

def get_file_content(repo_dir, file_path):
    full_path = os.path.join(repo_dir, file_path)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            lines = f.readlines()

        # Normalize blank lines with trailing whitespace
        lines = [(line.rstrip() + "\n") if line.strip() == "" else line for line in lines]

        content = "".join(lines)
        if not content.endswith("\n"):
            content += "\n"

        return content
    return None

def parse_test_info(instance):
    """Extract test file and test cases from FAIL_TO_PASS field."""
    if "FAIL_TO_PASS" not in instance:
        raise ValueError("No FAIL_TO_PASS field in instance data")
        
    # Parse the JSON string
    fail_to_pass = json.loads(instance["FAIL_TO_PASS"])
    
    # Extract unique test files and cases
    test_cases = []
    for test_path in fail_to_pass:
        parts = test_path.split("::")
        if len(parts) != 2:
            continue
        test_file, test_case = parts
        test_cases.append((test_file, test_case))
    
    if not test_cases:
        raise ValueError("No valid test cases found in FAIL_TO_PASS")
        
    return test_cases

def get_target_file(instance):
    if "FAIL_TO_PASS" in instance:
        fail_to_pass = json.loads(instance["FAIL_TO_PASS"])
        if len(fail_to_pass) > 0:
            return fail_to_pass[0].split("::")[0]
    raise ValueError("Cannot determine target file from FAIL_TO_PASS")

def extract_target_from_patch(patch_text):
    for line in patch_text.splitlines():
        if line.startswith('--- a/'):
            return line[6:].strip()
    return None

def main():
    instance = load_instance(INSTANCE_FILE, index=0)
    instance_id = instance["instance_id"]
    
    # First clone the repo to get the file content
    repo_dir = clone_repo(instance)
    try:
        # Get the file content first
        target_file = get_target_file(instance)
        file_content = get_file_content(repo_dir, target_file)
        file_content = file_content.rstrip() + "\n"

        if not file_content:
            raise ValueError("Could not find target file")
            
        # Generate patch with file content
        print(f"[DEBUG] Target file content for {target_file}:")
        print("\n".join(file_content.splitlines()[120:130]))

        print(f"[DEBUG] File line count: {target_file} -> {len(file_content.splitlines())}")

        target_path = os.path.join(repo_dir, target_file)
        with open(target_path, 'r') as f:
            lines = f.readlines()
        print(f"[DEBUG] {target_file} has {len(lines)} lines after checkout")

        code = generate_code_from_api(instance['problem_statement'], target_file, file_content)
        patch_target = extract_target_from_patch(code)
        if patch_target and not patch_target.endswith(target_file):
            print(f"[!] Warning: Model generated patch for unexpected file: {patch_target} vs expected: {target_file}")

        patch_path = save_patch(instance_id, code)
        print(f"Generated patch saved to {patch_path}")
        create_predictions_file(instance_id, patch_path)

        # Try to apply the patch
        try:
            # apply_patch(repo_dir, patch_path)
            # apply_unidiff_patch(patch_path, repo_dir)
            apply_unidiff_patch_relaxed(patch_path, repo_dir)
        except subprocess.CalledProcessError as e:
            print("Patch apply failed. Patch preview:")
            print(open(patch_path).read())
            raise

        # Get test cases from FAIL_TO_PASS
        test_cases = parse_test_info(instance)
        print(f"Running {len(test_cases)} test cases")
        
        all_results = []
        for test_file, test_case in test_cases:
            print(f"\nRunning test: {test_file}::{test_case}")
            
            generate_dockerfile(repo_dir, test_file, test_case)
            tag = f"local/sweb.eval.arm64.{instance_id}"
            build_docker_image(tag, repo_dir)
            stdout, stderr, code = run_docker_container(tag)
            
            all_results.append({
                "test_file": test_file,
                "test_case": test_case,
                "exit_code": code,
                "stdout": stdout,
                "stderr": stderr
            })

        result_path = os.path.join(RESULTS_DIR, f"result_{instance_id}.json")
        with open(result_path, "w") as f:
            json.dump({
                "instance_id": instance_id,
                "results": all_results
            }, f, indent=2)

        # Check if all tests passed
        if all(r["exit_code"] == 0 for r in all_results):
            print(f"[✓] All tests passed for {instance_id}")
        else:
            print(f"[✗] Some tests failed for {instance_id}")
            for r in all_results:
              print(f"  {r['test_file']}::{r['test_case']} - exit={r['exit_code']}")
              if r["exit_code"] != 0:
                  stderr_snippet = r["stderr"][:300].strip().replace('\n', ' ')
                  print(f"    stderr: {stderr_snippet}...")
                
    except Exception as e:
        print(f"[✗] Failed: {instance_id} – {e}")
    finally:
        cleanup_docker(f"local/sweb.eval.arm64.{instance_id}")
        shutil.rmtree(repo_dir)

if __name__ == "__main__":
    main()