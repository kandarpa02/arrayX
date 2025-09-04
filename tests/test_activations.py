import pytest
import torch
import torch.nn.functional as F
import nexnet

THRESHOLD = 2e-7

# Torch equivalents
ACTIVATIONS = {
    "relu": lambda t: F.relu(t),
    "relu6": lambda t: F.relu6(t),
    "leaky_relu": lambda t: F.leaky_relu(t, 0.01),
    "elu": lambda t: F.elu(t, 1.0),
    "gelu": lambda t: F.gelu(t, approximate="none"),
    "silu": lambda t: F.silu(t),
    "swish": lambda t: F.silu(t),
    "softplus": lambda t: F.softplus(t, beta=1.0, threshold=20.0),
    "softsign": lambda t: F.softsign(t),
    "hardtanh": lambda t: F.hardtanh(t, -1.0, 1.0),
    "tanh": lambda t: torch.tanh(t),
    "sigmoid": lambda t: torch.sigmoid(t),
    "logsigmoid": lambda t: F.logsigmoid(t),
    "softmax": lambda t: F.softmax(t, dim=-1),
    "mish": lambda t: F.mish(t),
}

# Neo equivalents
NEO_ACTS = {
    "relu": nexnet.nn.relu,
    "relu6": nexnet.nn.relu6,
    "leaky_relu": nexnet.nn.leaky_relu,
    "elu": nexnet.nn.elu,
    "gelu": nexnet.nn.gelu,
    "silu": nexnet.nn.silu,
    "swish": nexnet.nn.swish,
    "softplus": nexnet.nn.softplus,
    "softsign": nexnet.nn.softsign,
    "hardtanh": nexnet.nn.hardtanh,
    "tanh": nexnet.nn.tanh,
    "sigmoid": nexnet.nn.sigmoid,
    "logsigmoid": nexnet.nn.logsigmoid,
    "softmax": lambda t: nexnet.nn.softmax(t, dim=-1),
    "mish": nexnet.nn.mish,
}

SHAPES = [
    (3, 4), 
    (64, 64), 
    (128, 256)
]

errors_summary = {}

def run_test(act_name, shape):
    x_neo = nexnet.rand([shape[0], shape[1]], key=nexnet.RNGKey(0))
    y_neo = nexnet.rand([shape[1], shape[0]], key=nexnet.RNGKey(1))

    # Neo grads
    def f(x, y): 
        return NEO_ACTS[act_name](x @ y).sum()
    xg, yg = nexnet.grad(f)([x_neo, y_neo])

    # Torch grads
    torch.set_grad_enabled(True)
    x = x_neo.clone().requires_grad_()
    y = y_neo.clone().requires_grad_()
    out = ACTIVATIONS[act_name](x @ y).sum()
    out.backward()

    err_x = (xg - x.grad).abs().max().item()
    err_y = (yg - y.grad).abs().max().item()

    # record errors for summary
    errors_summary[(act_name, shape)] = (err_x, err_y)

    assert err_x < THRESHOLD, f"{act_name} x-grad mismatch (shape={shape}, err={err_x:.2e})"
    assert err_y < THRESHOLD, f"{act_name} y-grad mismatch (shape={shape}, err={err_y:.2e})"

@pytest.mark.parametrize("act_name", list(ACTIVATIONS.keys()))
@pytest.mark.parametrize("shape", SHAPES)
def test_activation(act_name, shape):
    run_test(act_name, shape)

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.section("Neo vs Torch Activation Grad Errors Summary")
    for (act_name, shape), (err_x, err_y) in errors_summary.items():
        terminalreporter.write_line(
            f"{act_name:12} | shape={shape} | max_err_x={err_x:.2e} | max_err_y={err_y:.2e}"
        )
