import requests
import json
import os
import re
import jsonlines
import subprocess
import tempfile
import shutil
import argparse
import difflib

API_URL = "http://localhost:8000/v1/code/completions"
HEADERS = {
    "Content-Type": "application/json"
}

MODEL = "polychat/2x/sonnet/sonnet_and_gpt4o"
# MODEL = "polychat/1x/sonnet"
PROMPT_TYPE = "code"
USER_ID = "1"
ONLY_CODE = True

PATCH_DIR = "my_patches"
RESULTS_DIR = "results"
DATASET_DIR = "data/swe-bench/"
INSTANCE_FILE = os.path.join(DATASET_DIR, "swe_bench_data.jsonl")

REPOS_DIR = "cloned_repos"

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
    prompt = f"""You are a helpful coding assistant.

You are working with the following file: `{target_file}`.

Below is the current content of the file:
<file_content>
{file_content}
</file_content>

--- 

Now here is a problem description:
{prompt_text}

Please modify the code above to fix the issue described.

IMPORTANT:
- Return the full, updated content of the file.
- Do not return a diff or patch.
- Do not add any explanation.
- Only output the complete, modified file content.
- Use exactly 4 spaces for indentation.
- Preserve all comments and formatting unless changes are required for the fix.

Begin your response below:
"""

    payload = {
        "model": MODEL,
        "stream": False,
        "user": USER_ID,
        "prompt_type": PROMPT_TYPE,
        "only_code": ONLY_CODE,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    # print(f"[DEBUG] Payload: {prompt}")

    print(f"[ℹ] Prompt created, requesting solution...")

    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
    response.raise_for_status()
    result = response.json()

    print(f"[✓] Received solution, returning...")

    return result["choices"][0]["message"]["content"]

def break_compilation_semantically(code: str) -> str:
    # Very naive way to break logic while keeping valid Python syntax
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if "return" in line:
            lines[ℹ] = line.replace("return", "return None  # forced fail")
            break
    return "\n".join(lines)

def clone_repo(instance):
    repo_name = instance['repo'].replace('/', '_')
    repo_dir = os.path.join(REPOS_DIR, repo_name)
    
    if not os.path.exists(repo_dir):
        raise ValueError(f"Repository {repo_name} not found in {REPOS_DIR}. Please run clone_repos.py first.")
    
    # Create a temporary copy of the repository to work with
    temp_dir = tempfile.mkdtemp(prefix="swebench_repo_")
    # Copy to a subdirectory with the repo name to maintain structure
    temp_repo_dir = os.path.join(temp_dir, repo_name)
    subprocess.run(["cp", "-r", repo_dir, temp_repo_dir], check=True)
    return temp_dir

def overwrite_target_file(repo_dir, file_path, new_code):
    # Get the repo name from the instance that was processed
    repo_name = [d for d in os.listdir(repo_dir) if not d.startswith('.')][0]
    # Construct full path including the repo name directory
    full_path = os.path.join(repo_dir, repo_name, file_path)
    with open(full_path, "w") as f:
        f.write(new_code)
    print(f"[✓] Overwrote file: {full_path}")

def create_code_patch(original_code: str, new_code: str, file_path: str) -> str:
    diff = difflib.unified_diff(
        original_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return ''.join(diff)

def get_repo_name(repo_dir):
    """Safely infer the repository folder name from the repo_dir."""
    candidates = [d for d in os.listdir(repo_dir) if os.path.isdir(os.path.join(repo_dir, d)) and not d.startswith('.')]
    if not candidates:
        raise RuntimeError(f"No valid subdirectory found in {repo_dir}")
    if len(candidates) > 1:
        print(f"[⚠️] Multiple directories found in {repo_dir}, using {candidates[0]}")
    return candidates[0]

def parse_test_info(instance, repo_dir):
    test_cases = []

    repo_name = get_repo_name(repo_dir)

    def process_test_paths(test_paths, test_type):
        results = []
        for test_path in test_paths:
            # Try to detect which format it is
            if "::" in test_path:
                # Old style: "tests/foo/test_bar.py::TestClass.test_method"
                test_file, test_case = test_path.split("::", 1)
                full_test_file_path = os.path.join(repo_dir, repo_name, test_file)

                if not os.path.exists(full_test_file_path):
                    print(f"[⚠️] Skipping {test_file}: file does not exist")
                    continue

                results.append((test_file, test_case, test_type))

            elif " (" in test_path and test_path.endswith(")"):
                # New style: "test_method (module.submodule.ClassName)"
                test_name, module_class = test_path[:-1].split(" (", 1)
                if "." not in module_class:
                    print(f"[⚠️] Skipping invalid module.class format: {test_path}")
                    continue

                module, class_name = module_class.rsplit(".", 1)

                # Build the file path
                test_file = module.replace(".", "/") + ".py"
                full_test_file_path = os.path.join(repo_dir, repo_name, test_file)

                if not os.path.exists(full_test_file_path):
                    full_test_file_path = os.path.join(repo_dir, repo_name, "tests", test_file)
                    if not os.path.exists(full_test_file_path):
                        print(f"[⚠️] Skipping {test_file}: file does not exist")
                        continue

                # Build the full test case
                test_case = f"{class_name}.{test_name}"

                results.append((test_file, test_case, test_type))

            else:
                print(f"[⚠️] Skipping unrecognized test format: {test_path}")
                continue

        return results

    if "FAIL_TO_PASS" in instance:
        print(f"[ℹ] Processing FAIL_TO_PASS tests")
        fail_to_pass = json.loads(instance["FAIL_TO_PASS"])
        print(f"[ℹ] Test paths: {fail_to_pass}")
        test_cases.extend(process_test_paths(fail_to_pass, "FAIL_TO_PASS"))

    if "PASS_TO_PASS" in instance:
        print(f"[ℹ] Processing PASS_TO_PASS tests")
        pass_to_pass = json.loads(instance["PASS_TO_PASS"])
        print(f"[ℹ] Test paths: {pass_to_pass}")
        test_cases.extend(process_test_paths(pass_to_pass, "PASS_TO_PASS"))

    if not test_cases:
        print(f"[⚠️] No valid test cases found for {instance['instance_id']}, skipping tests")

    return test_cases

def generate_dockerfile(
    repo_dir,
    test_file,
    test_case,
    include_test_patch=False,
    include_code_patch=False,
    patch_target_file=None,
    python_version="3.10"
):
    dockerfile = f"""\
# Base image: Python {python_version} (Debian-based slim for multi-arch arm64/x86_64 support)
FROM python:{python_version}-slim

# Avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONWARNINGS=ignore

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    git \\
    gfortran \\
    cmake \\
    ninja-build \\
    libopenblas-dev \\
    liblapack-dev \\
    libffi-dev \\
    libssl-dev \\
    libfreetype6-dev \\
    pkg-config \\
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and core tools
# RUN python -m pip install -U pip==23.2.1 setuptools==68.2.2 wheel==0.41.2

# scikit-learn specific
RUN python -m pip install -U \
    pip==23.2.1 \
    setuptools==68.2.2 \
    wheel==0.41.2 \
    flit-core==3.9.0

# 3.1 Pre-install common build dependencies (for PEP 518 builds) and test utilities
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    cython \
    setuptools_scm \
    extension-helpers \
    meson \
    meson-python \
    pybind11 \
    certifi \
    pytest \
    pytest-astropy \
    "asdf[tests]" \
    asdf-astropy

# 3.2 Install ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    update-ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Clone at env_setup_commit, then install
ARG REPO_URL
ARG ENV_SETUP_COMMIT
ARG BASE_COMMIT
WORKDIR /app
RUN git clone "$REPO_URL" repo && \\
    cd repo && git checkout "$ENV_SETUP_COMMIT"

# 5. Checkout base commit for patching + testing
WORKDIR /app/repo
RUN git checkout "$BASE_COMMIT"

# Optional: Patch setup.cfg for matplotlib
RUN if [ -f /app/repo/setup.cfg ]; then \\
      if ! grep -q '^\\[libs\\]' /app/repo/setup.cfg; then \\
        printf '\\n[libs]\\nlocal_freetype = True\\n' >> /app/repo/setup.cfg; \\
      elif ! grep -q '^local_freetype *= *True' /app/repo/setup.cfg; then \\
        sed -i '/^\\[libs\\]/a local_freetype = True' /app/repo/setup.cfg; \\
      fi; \\
    fi

# Install in editable mode
RUN pip install --no-cache-dir --no-build-isolation -e .

## ONLY INSTALL WHEN SKLEARN IS THE TARGET FILE
## Create a minimal setup.cfg if missing
## Work in clean dir
# WORKDIR /app

# Copy only the sklearn source folder
# RUN mkdir clean_sklearn && cp -r repo/sklearn clean_sklearn/sklearn

# Create minimal setup.py
# RUN echo "from setuptools import setup, find_packages; setup(name='scikit-learn', packages=find_packages())" > clean_sklearn/setup.py

# Install from clean version
# WORKDIR /app/clean_sklearn
# RUN pip install --no-cache-dir .

# Then continue whatever you were doing (patching, testing etc.)
"""

    if patch_target_file:
        dockerfile += f"""\
RUN echo '==== ORIGINAL FILE BEFORE PATCH ====' && \\
    cat /app/repo/{patch_target_file} || echo '[⚠️] Target file not found before patch'
"""

    if include_code_patch:
        dockerfile += """\
COPY code_patch.diff /app/code_patch.diff
RUN echo '==== APPLYING CODE PATCH ====' && \\
    cd /app/repo && git apply --reject --whitespace=fix /app/code_patch.diff || echo '[⚠️] Failed to apply code patch'
"""

    if include_test_patch:
        dockerfile += """\
COPY test_patch.diff /app/test_patch.diff
RUN echo '==== APPLYING TEST PATCH ====' && \\
    cd /app/repo && git apply --reject --whitespace=fix /app/test_patch.diff || echo '[⚠️] Failed to apply test patch'
"""

    if patch_target_file:
        dockerfile += f"""\
RUN echo '==== FILE AFTER PATCH ====' && \\
    cat /app/repo/{patch_target_file} || echo '[⚠️] Target file not found after patch'
"""

    dockerfile += """
# Final working directory
WORKDIR /app/repo
CMD ["pytest"]
"""

    with open(os.path.join(repo_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile)

def build_docker_image(tag, context_dir, instance, extra_files=None):
    """
    Build a Docker image with optional patch files.

    Args:
        tag (str): Docker image tag.
        context_dir (str): Directory containing Dockerfile and optional patches.
        repo_url (str): GitHub repository URL.
        repo_commit (str): Commit hash to check out in the image.
        extra_files (list): Optional list of filenames (within context_dir) to include in build context.
    """
    # Collect files to copy into build context
    dockerfile_path = os.path.join(context_dir, "Dockerfile")
    files_to_copy = [dockerfile_path]

    if extra_files:
        for fname in extra_files:
            fpath = os.path.join(context_dir, fname)
            if os.path.exists(fpath):
                files_to_copy.append(fpath)
            else:
                print(f"[⚠️] Skipping missing patch file: {fname}")

    print(f"[ℹ] Files being copied into Docker build context: {files_to_copy}")

    # Create temporary build context and copy files
    build_context = tempfile.mkdtemp()
    for file in files_to_copy:
        shutil.copy(file, os.path.join(build_context, os.path.basename(file)))

    # Run Docker build
    # subprocess.run([
    #     "docker", "build",
    #     "--build-arg", f"REPO_URL={repo_url}",
    #     "--build-arg", f"REPO_COMMIT={repo_commit}",
    #     "-t", tag,
    #     build_context
    # ], check=True)
    
    subprocess.run([
        "docker", "build",
        "--build-arg", f"REPO_URL=https://github.com/{instance['repo']}.git",
        "--build-arg", f"ENV_SETUP_COMMIT={instance.get('environment_setup_commit', instance['base_commit'])}",
        "--build-arg", f"BASE_COMMIT={instance['base_commit']}",
        "-t", tag,
        build_context
    ], check=True)

    shutil.rmtree(build_context)

def run_docker_container(tag, test_file=None, test_case=None, collect_only=False):
    cmd = [
        "docker", "run", "--rm", tag,
        "pytest", "-vv",
        "-p", "no:warnings",
        "-o", "filterwarnings=ignore::DeprecationWarning"
    ]

    if collect_only:
        cmd.append("--collect-only")
    elif test_file and test_case:
        cmd.append(f"{test_file}::{test_case}")
    elif test_file:
        cmd.append(test_file)

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Save full output for debugging
    with open("debug_stdout.txt", "w") as f:
        f.write(result.stdout)
    with open("debug_stderr.txt", "w") as f:
        f.write(result.stderr)

    return result.stdout, result.stderr, result.returncode

def extract_file_from_docker_image(image_tag, internal_path, output_path):
    """
    Copy a file from inside a built Docker image to the host filesystem.
    """
    container_name = f"tmp_extract_{image_tag.replace('/', '_')}"
    try:
        # Create container from image (don't start)
        subprocess.run(["docker", "create", "--name", container_name, image_tag], check=True)

        # Copy the file from image
        subprocess.run(["docker", "cp", f"{container_name}:{internal_path}", output_path], check=True)
        print(f"[✓] Extracted file from Docker image: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[✗] Failed to extract file: {e}")
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def docker_image_exists(image_tag, timeout=5):
    """Check if a Docker image exists locally using `docker image inspect`."""
    try:
        print(f"[ℹ] Checking if Docker image '{image_tag}' exists...")
        result = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[⚠️] Timeout while checking Docker image '{image_tag}'")
        return False
    except subprocess.SubprocessError as e:
        print(f"[⚠️] Failed to check Docker image '{image_tag}': {e}")
        return False

def cleanup_docker(tag):
    if docker_image_exists(tag):
        print(f"[ℹ] Removing docker image {tag}")
        subprocess.run(["docker", "rmi", "-f", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"[ℹ] Docker image {tag} does not exist, skipping cleanup")

def get_file_content(repo_dir, file_path):
    # Get the repo name from the instance that was processed
    repo_name = [d for d in os.listdir(repo_dir) if not d.startswith('.')][0]
    # Construct full path including the repo name directory
    full_path = os.path.join(repo_dir, repo_name, file_path)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            lines = f.readlines()

        lines = [(line.rstrip() + "\n") if line.strip() == "" else line for line in lines]
        content = "".join(lines)
        if not content.endswith("\n"):
            content += "\n"
        return content
    return None

def get_target_file(instance):
    # Prefer patch over test_patch because it's the real code fix
    patch_text = instance.get("patch") or instance.get("test_patch")
    if patch_text:
        match = re.search(r"^diff --git a/(\S+) b/", patch_text, re.MULTILINE)
        if match:
            return match.group(1)

    raise ValueError("Cannot determine target file from patch or test_patch")

def save_original_code(instance_id, code):
    code_path = os.path.join(PATCH_DIR, f"{instance_id}.original.py")
    with open(code_path, "w") as f:
        f.write(code)
    return code_path

def save_patch(instance_id, code):
    patch_path = os.path.join(PATCH_DIR, f"{instance_id}.patch")
    with open(patch_path, "w") as f:
        f.write(code)
    return patch_path

def count_instances(instance_file):
    count = 0
    with jsonlines.open(instance_file) as reader:
        for _ in reader:
            count += 1
    return count

def find_instance_by_id(instance_file, target_id):
    with jsonlines.open(instance_file) as reader:
        for item in reader:
            if item["instance_id"] == target_id:
                return item
    return None

def is_test_defined_in_patch(instance):
    """
    Returns True if FAIL_TO_PASS test appears to be introduced in test_patch.
    """
    if "test_patch" not in instance or "FAIL_TO_PASS" not in instance:
        return False

    fail_to_pass = json.loads(instance["FAIL_TO_PASS"])
    test_patch = instance["test_patch"]

    # Check if any test file in FAIL_TO_PASS appears in the patch
    for test_entry in fail_to_pass:
        test_file = test_entry.split("::")[0]
        if f"diff --git a/{test_file}" in test_patch:
            return True
    return False

def apply_test_patch(repo_dir, instance):
    """
    Applies the test_patch to the repo using `patch` (fuzz-tolerant).
    Only test-related files from the patch are applied.
    """
    if "test_patch" not in instance:
        return

    repo_name = [d for d in os.listdir(repo_dir) if not d.startswith('.')][0]
    full_repo_path = os.path.join(repo_dir, repo_name)

    # Extract only test-related hunks
    test_patch_lines = []
    inside_test_file = False
    for line in instance["test_patch"].splitlines():
        if line.startswith("diff --git"):
            inside_test_file = "/tests/" in line
        if inside_test_file:
            test_patch_lines.append(line)

    if not test_patch_lines:
        print("[⚠️] No test hunks found in test_patch")
        return

    patch_text = "\n".join(test_patch_lines) + "\n"

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(patch_text)
        tmp_path = tmp.name

    try:
        subprocess.run(
            ["patch", "-p1", "--fuzz=3", "-i", tmp_path],
            cwd=full_repo_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"[✓] Applied test patch to {repo_name} with fuzz")
    except subprocess.CalledProcessError as e:
        print(f"STDOUT:\n{e.stdout.decode().strip()}")
        print(f"STDERR:\n{e.stderr.decode().strip()}")
    finally:
        os.unlink(tmp_path)

def safe_remove_dir(dir_path):
    """Safely remove a directory if it exists, silently ignore if it doesn't."""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    except Exception as e:
        print(f"[⚠️] Failed to remove directory {dir_path}: {e}")

def process_instance(instance, instance_id):
    print(f"\nProcessing instance: {instance_id}")
    print(f"[ℹ] Checking out commit {instance['base_commit']} from repo {instance['repo']}")

    repo_dir = clone_repo(instance)
    print(f"[✓] Cloned repo {instance['repo']} to {repo_dir}")

    try:
        repo_name = get_repo_name(repo_dir)
        target_file = get_target_file(instance)
        file_content = get_file_content(repo_dir, target_file)

        if file_content is None:
            print(f"[⚠️] Could not find target file {target_file} in repo {instance['repo']}")
            errror_results = []
            errror_results.append({
                "test_file": target_file,
                "test_case": "None",
                "test_type": "None",
                "exit_code": 5,
                "stdout": "",
                "stderr": f"Could not find target file {target_file} in repo {instance['repo']}"
            })

            error_results_path = os.path.join(RESULTS_DIR, f"result_{instance_id}.json")
            with open(error_results_path, "w") as f:
                json.dump({
                    "instance_id": instance_id,
                    "results": errror_results
                }, f, indent=2)
            
            safe_remove_dir(repo_dir)
            return True

        file_content = file_content.rstrip() + "\n"

        if not file_content:
            raise ValueError("Could not find target file")

        # Save canonical patch from instance["patch"]
        # Uncomment to test if patch works with benchmark provided patch
        if "patch" not in instance or not instance["patch"].strip():
            raise ValueError(f"[✗] Instance {instance_id} is missing a 'patch' field.")

        code_patch_path = os.path.join(repo_dir, "code_patch.diff")
        with open(code_patch_path, "w") as f:
            f.write(instance["patch"])
        print(f"[✓] Canonical patch written to {code_patch_path}")
        print(f"[✓] Patch created for {instance_id} and saved to {code_patch_path}")
        
        test_cases = parse_test_info(instance, repo_dir)
        print(f"Running {len(test_cases)} test cases")

        # Save the test_patch to disk (if relevant)
        if is_test_defined_in_patch(instance):
            print("[ℹ] FAIL_TO_PASS test appears to be defined in patch; will apply at container runtime...")
            patch_file_path = os.path.join(repo_dir, "test_patch.diff")
            with open(patch_file_path, "w") as f:
                f.write(instance["test_patch"])
        else:
            patch_file_path = None
        
        all_results = []
        test_index = 0
        for test_file, test_case, test_type in test_cases:
            print(f"\nRunning test {test_index + 1}/{len(test_cases)}: {test_file}::{test_case}")

            generate_dockerfile(
                repo_dir,
                test_file,
                test_case,
                include_test_patch=(patch_file_path is not None),
                include_code_patch=True,
                patch_target_file=target_file,
                python_version="3.9" if instance.get("version") in ["1.3", "2.4"] else "3.10"
            )
            tag = f"local/sweb.eval.arm64.{instance_id}"
            
            extra_files = []
            if patch_file_path:
                extra_files.append("test_patch.diff")
            if code_patch_path:
                extra_files.append("code_patch.diff")

            build_docker_image(tag, repo_dir, instance, extra_files=extra_files)

            stdout, stderr, code = run_docker_container(tag, test_file, test_case, collect_only=False)

            all_results.append({
                "test_file": test_file,
                "test_case": test_case,
                "test_type": test_type,
                "exit_code": code,
                "stdout": stdout,
                "stderr": stderr
            })

            test_index += 1

        result_path = os.path.join(RESULTS_DIR, f"result_{instance_id}.json")
        with open(result_path, "w") as f:
            json.dump({
                "instance_id": instance_id,
                "results": all_results
            }, f, indent=2)

        if all(r["exit_code"] == 0 for r in all_results):
            print(f"✅ All tests passed for {instance_id}")
        else:
            print(f"Some tests failed for {instance_id}")
            non_passed_results = [r for r in all_results if r["exit_code"] == 1]
            failed_results = [r for r in all_results if r["exit_code"] != 0 and r["exit_code"] != 1]
            for r in all_results:
                print(f"  {r['test_file']}::{r['test_case']} - exit={r['exit_code']}")
                if r["exit_code"] == 1:
                    stderr_snippet = r["stderr"][:300].strip().replace('\n', ' ')
                    non_passed_results.append(r)
                    print(f"    stderr: {stderr_snippet}...")
                elif r["exit_code"] != 0:
                    stdout_snippet = r["stdout"][:300].strip().replace('\n', ' ')
                    failed_results.append(r)
                    print(f"    stdout: {stdout_snippet}...")
            
            if len(non_passed_results) > 0:
                print(f"❌ Some tests didn't pass for {instance_id}")
                for r in non_passed_results:
                    print(f"  {r['test_file']}::{r['test_case']} - exit={r['exit_code']}")
                    stderr_snippet = r["stderr"][:300].strip().replace('\n', ' ')
                    print(f"    stderr: {stderr_snippet}...")

            if len(failed_results) > 0:
                print(f"⚠️ Some tests failed for {instance_id}")
                for r in failed_results:
                    print(f"  {r['test_file']}::{r['test_case']} - exit={r['exit_code']}")
                    stdout_snippet = r["stdout"][:300].strip().replace('\n', ' ')
                    print(f"    stdout: {stdout_snippet}...")

    except Exception as e:
        print(f"[✗] Failed: {instance_id} – {e}")
        safe_remove_dir(repo_dir)
        return False
    finally:
        print(f"[ℹ] Cleaning up docker image for {instance_id}")
        cleanup_docker(f"local/sweb.eval.arm64.{instance_id}")
        safe_remove_dir(repo_dir)
    
    print(f"[✓] Successfully processed {instance_id}")
    return True

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run SWE-bench tests')
    parser.add_argument('--instance-id', type=str, help='Specific instance ID to run')
    parser.add_argument('--limit', type=int, help='Limit the number of benchmarks to run')
    parser.add_argument('--start-instance-id', type=str, help='Start processing from this instance ID forward')
    args = parser.parse_args()

    if args.instance_id:
        # Find the instance and its index
        target_instance = find_instance_by_id(INSTANCE_FILE, args.instance_id)
        if not target_instance:
            print(f"Error: Instance ID {args.instance_id} not found")
            return

        # Find the index of the target instance
        target_index = 0
        with jsonlines.open(INSTANCE_FILE) as reader:
            for i, instance in enumerate(reader):
                if instance["instance_id"] == args.instance_id:
                    target_index = i
                    break

        # Run the target instance
        process_instance(target_instance, args.instance_id)

        # If limit is specified, run additional instances
        if args.limit:
            print(f"\nRunning {args.limit - 1} additional instances after {args.instance_id}")
            index = target_index + 1
            count = 1  # Start at 1 since we've already processed the target instance

            while count < args.limit:
                try:
                    instance = load_instance(INSTANCE_FILE, index=index)
                    instance_id = instance["instance_id"]
                    
                    if not process_instance(instance, instance_id):
                        break
                    
                    index += 1
                    count += 1
                except IndexError:
                    print(f"\nReached end of instances at index {index}")
                    break
    else:
        # Original behavior - run all instances
        total_instances = count_instances(INSTANCE_FILE)
        if args.limit:
            total_instances = min(total_instances, args.limit)
            print(f"\nFound {total_instances} instances to process (limited from {count_instances(INSTANCE_FILE)})")
        else:
            print(f"\nFound {total_instances} instances to process")

        # Find starting index if start_instance_id is provided
        start_index = 0
        if args.start_instance_id:
            with jsonlines.open(INSTANCE_FILE) as reader:
                for i, instance in enumerate(reader):
                    if instance["instance_id"] == args.start_instance_id:
                        start_index = i
                        print(f"\nStarting from instance ID {args.start_instance_id} at index {start_index}")
                        break
                else:
                    print(f"Error: Start instance ID {args.start_instance_id} not found")
                    return

        index = start_index
        while True:
            try:
                instance = load_instance(INSTANCE_FILE, index=index)
                instance_id = instance["instance_id"]
                
                if not process_instance(instance, instance_id):
                    break
                
                index += 1
                if args.limit and index >= args.limit:
                    print(f"\nReached limit of {args.limit} instances")
                    break
            except IndexError:
                print(f"\nReached end of instances at index {index}")
                break

    # Generate report with timestamp
    print(f"\nGenerating report")
    subprocess.run(["python", "generate_report.py"], check=True)
    print(f"Report generated in /reports")

if __name__ == "__main__":
    main()
