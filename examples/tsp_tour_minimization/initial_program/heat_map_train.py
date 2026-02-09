import pathlib

BASE_DIR = pathlib.Path(__file__).parent

# torch & related imports
import numpy as np
import torch


# You can train graph neural networks here (actually you can edit the whole file).
# Save them in str(BASE_DIR / "pretrained") directory and use them in `heat_map_inference.py`.
# But mind the training time, it should not exceed 6 minutes (python3.11, ubuntu 22.04, nvidia A100 40 GB GPU).

# In the original UTSP paper the heat map matrix is used, but upon the closer look on the authors code, they did not use it (they used simple K nearest neighbours as candidates).
# So, the SOTA metrics were achieved without a heat map matrix, but maybe GNN approach is not wrong by its nature, maybe if trained properly it can help 2'opt and k'opt algorithms to find the best solution faster.

# Possible GNN step (just in case, implement if you like): hamiltonian cycle constraint, loss on exact 2 degree for each node, etc.


if __name__ == "__main__":
    print("Sample output to log")
