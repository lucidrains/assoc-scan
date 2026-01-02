## assoc-scan

A minimal Pytorch implementation of an associative scan, with support for accelerated backends.

## Install

```bash
$ pip install assoc-scan
```

## Usage

```python
import torch
from assoc_scan import AssocScan

scan = AssocScan()

gates = torch.randn(1, 1024, 512).sigmoid()
inputs = torch.randn(1, 1024, 512)

out = scan(gates, inputs) # (1, 1024, 512)
```

To use the accelerated version, simply pass `use_accelerated = True` and ensure [accelerated-scan](https://github.com/proger/accelerated-scan) is installed.

```python
scan = AssocScan(use_accelerated = True)

out = scan(gates, inputs)
```

## Citations

```bibtex
@software{Kyrylov_Accelerated_Scan_2024,
    author    = {Kyrylov, Volodymyr},
    doi       = {10.5281/zenodo.10600962},
    month     = {jan},
    title     = {{Accelerated Scan}},
    version   = {0.1.2},
    year      = {2024}
}
```
