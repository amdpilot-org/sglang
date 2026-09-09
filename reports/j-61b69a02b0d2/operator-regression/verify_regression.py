"""Operator-owned GPU red/green check of the generated NaN regression."""
import difflib
import json
import math
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path('/job')
BASE = '817319afcdf93e6830bc1e3ac8d63b9dd6b0df73'
HEAD = 'fba24b05adea7a73150df11b6fc84322f38dcec3'
TEST = 'test/registered/unit/mem_cache/test_mha_fp8_kv_write_hip.py'
NODE = TEST + '::TestMHAFP8KVWriteHIP::test_fp16_zero_scale_nan_payload'
PYTHON = '/opt/venv/bin/python'


def run(args, *, cwd=ROOT, env=None, name=None, expected=0):
    proc = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=240)
    if name:
        (ROOT / (name + '.log')).write_text(proc.stdout)
    if proc.returncode != expected:
        raise RuntimeError(f'{name or args[0]} returned {proc.returncode}: {proc.stdout[-4000:]}')
    return proc.stdout


run(['git', 'clone', '--depth=1', '--no-checkout', 'https://github.com/amdpilot-org/sglang.git', 'control'])
control = ROOT / 'control'
run(['git', 'fetch', '--depth=1', 'origin', BASE], cwd=control)
run(['git', 'checkout', '--detach', BASE], cwd=control)
assert run(['git', 'rev-parse', 'HEAD'], cwd=control).strip() == BASE
url = f'https://raw.githubusercontent.com/amdpilot-org/sglang/{HEAD}/reports/j-61b69a02b0d2/pr55-fp16-nan-payload.patch'
with urllib.request.urlopen(url, timeout=30) as response:
    patch = response.read().decode()
original_line = '+            [1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0],'
fixed_line = '+            [0.0, -0.0, float("nan"), float("nan"), 1.0, -1.0, 8.0, -8.0],'
assert patch.count(original_line) == 1
(ROOT / 'original.patch').write_text(patch)
(ROOT / 'corrected.patch').write_text(patch.replace(original_line, fixed_line))
run(['git', 'apply', '--include=' + TEST, str(ROOT / 'original.patch')], cwd=control)


def test_env(checkout, cache):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(checkout / 'python'),
               TRITON_CACHE_DIR='/tmp/pr63-regression/' + cache,
               TORCHINDUCTOR_CACHE_DIR='/tmp/pr63-regression/inductor-' + cache)
    return env


baseline_env = test_env(control, 'baseline')
run([PYTHON, '-m', 'pytest', '-q', NODE], cwd=control, env=baseline_env,
    name='original-test-on-baseline', expected=0)
test = control / TEST
text = test.read_text()
assert text.count(original_line[1:]) == 1
text = text.replace(original_line[1:], fixed_line[1:])
for tensor in ('k', 'v'):
    old = f'self.assertTrue(torch.equal({tensor}, {tensor}_before))'
    new = f'self.assertTrue(torch.equal({tensor}.view(torch.uint8), {tensor}_before.view(torch.uint8)))'
    assert text.count(old) == 1
    text = text.replace(old, new)
test.write_text(text)
run([PYTHON, '-m', 'pytest', '-q', NODE], cwd=control, env=baseline_env,
    name='corrected-test-on-baseline', expected=1)
run(['git', 'clone', '--no-hardlinks', str(control), 'candidate'])
candidate = ROOT / 'candidate'
assert run(['git', 'rev-parse', 'HEAD'], cwd=candidate).strip() == BASE
run(['git', 'apply', str(ROOT / 'corrected.patch')], cwd=candidate)
test = candidate / TEST
text = test.read_text()
for tensor in ('k', 'v'):
    old = f'self.assertTrue(torch.equal({tensor}, {tensor}_before))'
    new = f'self.assertTrue(torch.equal({tensor}.view(torch.uint8), {tensor}_before.view(torch.uint8)))'
    assert text.count(old) == 1
    text = text.replace(old, new)
test.write_text(text)
run([PYTHON, '-m', 'pytest', '-q', NODE], cwd=candidate,
    env=test_env(candidate, 'candidate'), name='corrected-test-on-candidate', expected=0)
run([PYTHON, '-m', 'pytest', '-q', TEST], cwd=candidate,
    env=test_env(candidate, 'candidate'), name='candidate-neighbors', expected=0)
import torch
import triton
assert torch.cuda.device_count() == 1
decoded = torch.tensor([0xFF, 0x80], dtype=torch.uint8).view(torch.float8_e4m3fnuz).float().tolist()
assert decoded[0] == -240.0 and math.isnan(decoded[1])
report = {
    'base': BASE, 'report_head': HEAD, 'visible_gpus': torch.cuda.device_count(),
    'gpu_name': torch.cuda.get_device_name(0), 'hip': torch.version.hip,
    'torch': torch.__version__, 'triton': triton.__version__,
    'original_regression_passes_unfixed_kernel': True,
    'corrected_regression_fails_unfixed_kernel': True,
    'corrected_regression_passes_candidate': True,
    'fnuz_ff_decodes_to': -240.0, 'fnuz_80_decodes_to': 'NaN',
    'conclusion': 'NaN became a finite value; this is not merely a NaN payload difference.',
}
(ROOT / 'regression-proof.json').write_text(json.dumps(report, indent=2) + '\n')
out = run(['git', 'diff', '--binary', '--full-index', BASE], cwd=candidate)
(ROOT / 'recovery.patch').write_text(out)
(ROOT / 'corrected.patch').write_text(out)
(ROOT / 'regression-proof.patch').write_text(''.join(difflib.unified_diff(
    [], (ROOT / 'regression-proof.json').read_text().splitlines(keepends=True),
    fromfile='/dev/null', tofile='b/regression-proof.json')))
print(json.dumps(report))
