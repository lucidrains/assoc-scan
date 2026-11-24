import torch
import pytest
from assoc_scan import AssocScan

def test_assoc_scan_basic():
    scan = AssocScan()
    gates = torch.randn(2, 16, 8).sigmoid()
    inputs = torch.randn(2, 16, 8)

    out = scan(gates, inputs)
    assert out.shape == (2, 16, 8)

def test_assoc_scan_reverse():
    scan = AssocScan(reverse = True)
    gates = torch.randn(1, 10, 4).sigmoid()
    inputs = torch.randn(1, 10, 4)

    out = scan(gates, inputs)
    assert out.shape == (1, 10, 4)

def test_assoc_scan_no_batch():
    scan = AssocScan()
    gates = torch.randn(10, 4).sigmoid()
    inputs = torch.randn(10, 4)

    out = scan(gates, inputs)
    assert out.shape == (10, 4)

def test_assoc_scan_prev():
    scan = AssocScan()
    gates = torch.randn(1, 10, 4).sigmoid()
    inputs = torch.randn(1, 10, 4)
    prev = torch.randn(1, 4)

    out = scan(gates, inputs, prev = prev)
    assert out.shape == (1, 10, 4)

@pytest.mark.parametrize("use_accelerated", [False]) # Can't test accelerated easily without dependencies/GPU
def test_assoc_scan_accelerated_flag(use_accelerated):
    scan = AssocScan(use_accelerated = use_accelerated)
    gates = torch.randn(1, 16, 8).sigmoid()
    inputs = torch.randn(1, 16, 8)

    out = scan(gates, inputs)
    assert out.shape == (1, 16, 8)

@pytest.mark.parametrize("has_prev", [False, True])
def test_assoc_scan_reverse_matches_flip(has_prev):
    gates = torch.randn(2, 17, 4).sigmoid()
    inputs = torch.randn(2, 17, 4)
    prev = torch.randn(2, 4) if has_prev else None

    out = AssocScan(reverse = True)(gates, inputs, prev = prev)
    expected = AssocScan()(gates.flip(1), inputs.flip(1), prev = prev).flip(1)

    assert torch.allclose(out, expected, atol = 1e-6)

def test_assoc_scan_prev_no_feature_dim():
    scan = AssocScan(reverse = True, has_no_feature_dim = True)
    gates = torch.randn(2, 10).sigmoid()
    inputs = torch.randn(2, 10)
    prev = torch.randn(2)

    out = scan(gates, inputs, prev = prev)
    assert out.shape == (2, 10)

    prev_2d = torch.randn(2, 1)
    out_2d = scan(gates, inputs, prev = prev_2d)
    assert out_2d.shape == (2, 10)
