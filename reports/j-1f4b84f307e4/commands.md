# Commands run

All commands below were run in the supplied container. Long inspection commands
are abbreviated to their principal operations where repeated `sed`/`rg` calls
only read source.

```bash
# Record environment, GPU, Torch/ROCm, git, gh, and disk identity.
env; date; /opt/venv/bin/python - <<'PY'
import sys, torch
print(sys.version)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
rocm-smi --showproductname --showdriverversion
git --version
gh --version
df -h /job /tmp

# Clone the delivery mirror with bounded retries.
git clone --depth=200 https://github.com/amdpilot-org/sglang.git /job/sglang
cd /job/sglang
git remote -v
git rev-parse HEAD
git branch --show-current
git status --short

# Read issue 34367, its timeline, and nearby PR search results.
gh issue view 34367 --repo sgl-project/sglang --json title,state,author,createdAt,updatedAt,body,labels,url,comments
gh issue view 34367 --repo sgl-project/sglang --comments
gh api --paginate repos/sgl-project/sglang/issues/34367/timeline --jq '...'
gh search prs "LongLive2" --repo sgl-project/sglang --json number,title,state,url,createdAt,updatedAt,isDraft --limit 50
gh search prs "num_frames_per_block causal denoising" --repo sgl-project/sglang --json number,title,state,url,createdAt,updatedAt,isDraft --limit 50
gh search issues "LongLive2 num_frames_per_block" --repo sgl-project/sglang --json number,title,state,url,isPullRequest,createdAt,updatedAt --limit 20

# Locate and inspect LongLive2 config, request admission, and causal stage source.
rg -n -i "longlive|causal.?denois|num_frames_per_block" python/sglang/multimodal_gen
git log --date=iso-strict --format='%H %ad %an %s' -- <relevant files>
git blame -L 35,75 python/sglang/multimodal_gen/configs/pipeline_configs/longlive2.py
gh pr view 38226 --repo sgl-project/sglang --json number,title,state,mergedAt,baseRefName,headRefName,headRepositoryOwner,commits,body,url
gh pr view 27639 --repo sgl-project/sglang --json number,title,state,mergedAt,baseRefName,headRefName,headRepositoryOwner,body,url
sed/rg source inspection of longlive2.py, causal_denoising.py, base.py, schedule_batch.py, server_args.py, and existing unit tests

# Add the synthetic GPU admission test with apply_patch.
apply_patch <<'PATCH'
<test file patch>
PATCH

# Initial focused run (found two test-only shape expectations).
export PYTHONPATH=/job/sglang/python
/opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py -vv

# Correct block-shape expectations and rerun.
apply_patch <<'PATCH'
<expectation correction>
PATCH
export PYTHONPATH=/job/sglang/python
/opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py -vv

# Run existing config coverage plus the new test on current main.
export PYTHONPATH=/job/sglang/python
/opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_longlive2_pipeline_config.py \
  python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py -vv

# Fetch read-only PR 38226 head and test it in a detached worktree.
git fetch --depth=1 https://github.com/sgl-project/sglang.git pull/38226/head
git worktree add --detach /job/sglang-pr38226 FETCH_HEAD
cp /job/sglang/python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py \
  /job/sglang-pr38226/python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py
cd /job/sglang-pr38226
export PYTHONPATH=/job/sglang-pr38226/python
/opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_longlive2_pipeline_config.py \
  python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py -vv

# Check formatter availability and repository state.
/opt/venv/bin/ruff --version
/opt/venv/bin/ruff check <test file>
/opt/venv/bin/ruff format --check <test file>
git status --short
git diff -- <test file>

# Save final raw pytest logs.
mkdir -p reports/j-1f4b84f307e4
export PYTHONPATH=/job/sglang/python
/opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_longlive2_pipeline_config.py \
  python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py \
  -vv 2>&1 | tee reports/j-1f4b84f307e4/pytest-current-main.log
cd /job/sglang-pr38226
export PYTHONPATH=/job/sglang-pr38226/python
/opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_longlive2_pipeline_config.py \
  python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py \
  -vv 2>&1 | tee /job/sglang/reports/j-1f4b84f307e4/pytest-pr-38226.log

# Create the required delivery branch, commit, push, and open a draft PR.
git checkout -b amdpilot/j-1f4b84f307e4
git add python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py \
  reports/j-1f4b84f307e4
git add -f reports/j-1f4b84f307e4/pytest-current-main.log \
  reports/j-1f4b84f307e4/pytest-pr-38226.log
git commit -m "test: validate LongLive2 causal frame admission"
git push -u origin amdpilot/j-1f4b84f307e4
git push --force-with-lease=refs/heads/amdpilot/j-1f4b84f307e4:3208edbb37ac2234ad953abc5b817eb1e6b3ac2e \
  origin amdpilot/j-1f4b84f307e4
gh pr create --repo amdpilot-org/sglang \
  --base main --head amdpilot/j-1f4b84f307e4 --draft \
  --title "test: validate LongLive2 causal frame admission" \
  --body "<PR body summarized in README.md>"
```
