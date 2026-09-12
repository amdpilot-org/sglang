import torch


assert torch.cuda.is_available()
torch.manual_seed(29960)
x = torch.linspace(-1, 1, 4096, device="cuda:0", dtype=torch.float32)
static = x.clone()

# Warm the exact operation sequence before capture.
y = static
for _ in range(256):
    y = y * 1.0001 + 0.0001
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = static
    for _ in range(256):
        y = y * 1.0001 + 0.0001

static.copy_(x)
graph.replay()
torch.cuda.synchronize()

# Independent closed-form reference in float64:
# x_n = a**n * x_0 + (a**n - 1), because b == a - 1 == 0.0001.
scale = 1.0001**256
reference = x.double() * scale + (scale - 1.0)
max_abs = (y.double() - reference).abs().max().item()

print("device_name=", torch.cuda.get_device_name(0))
print("gcn_arch=", torch.cuda.get_device_properties(0).gcnArchName)
print("torch=", torch.__version__, "hip=", torch.version.hip, "cuda=", torch.version.cuda)
print("graph_type=", type(graph).__name__, "ops_in_fixture=512")
print("max_abs_vs_float64_closed_form=", format(max_abs, ".9g"))
print("allclose_atol_2e-5=", torch.allclose(y.double(), reference, rtol=2e-5, atol=2e-5))

# Boundary case: replay must consume changed static input without recapture.
static.fill_(0.25)
graph.replay()
torch.cuda.synchronize()
reference_mutated = torch.full_like(x.double(), 0.25) * scale + (scale - 1.0)
print(
    "mutated_input_max_abs=",
    format((y.double() - reference_mutated).abs().max().item(), ".9g"),
)
