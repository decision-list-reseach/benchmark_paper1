<div align="center">

# CORELS Installation Guide

**Certifiably Optimal Rule Lists (CORELS)**

Python installation guide for this research project.

</div>

---

## Overview

CORELS (Certifiably Optimal RulE ListS) is a C++-backed machine learning library that produces provably optimal and highly interpretable rule lists. Because it depends on native C++ extensions, installing CORELS requires a few additional steps compared to a typical Python package.

---

## Important

> **Python 3.10 is required.**
>
> Python **3.11+** removed the `longintrepr.h` header from the public CPython API.
> The CORELS C++ extension depends on this header and therefore **cannot be compiled** on newer Python versions without modifying the source code.

---

## Note

> The official **PyPI** release of `corels` is currently unmaintained.
>
> For this project we recommend installing the actively maintained community fork:
>
> **https://github.com/fingoldin/pycorels**

---

# Step 1 — Install System Dependencies

CORELS requires the **GNU Multiple Precision Arithmetic Library (GMP)**.

### macOS

```bash
brew install python@3.10
brew install gmp
```

### Ubuntu / Debian

```bash
sudo apt update

sudo apt install \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    libgmp-dev
```

### Windows

> Native Windows compilation is not recommended.
>
> Please install **WSL2 (Ubuntu)** and follow the Linux instructions instead.
>
> https://learn.microsoft.com/windows/wsl/install

---

# Step 2 — Create a Virtual Environment

```bash
python3.10 -m venv .venv_corels

source .venv_corels/bin/activate
```

---

# Step 3 — Install CORELS

```bash
git clone https://github.com/fingoldin/pycorels.git

cd pycorels

pip install .
```

---

## Tip

After the installation completes successfully, the cloned source repository is no longer required.

You may safely remove it:

```bash
cd ..

rm -rf pycorels
```

---

# Step 4 — Install Remaining Packages

```bash
pip install pandas scikit-learn scipy numpy
```

---

# Step 5 — Verify Installation

```bash
python -c "from corels import CorelsClassifier; print('CORELS installed successfully!')"
```

If you see:

```text
CORELS installed successfully!
```

the installation completed correctly.
