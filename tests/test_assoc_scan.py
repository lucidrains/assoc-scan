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
