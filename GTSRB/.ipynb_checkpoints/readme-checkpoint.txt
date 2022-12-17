Run AugTest.py for all transformation (6 types):

Parameters contains:
model: choice [0, 1, 2]
decay: float, 1.0
step_0-step_5 : float, 0.1
max_search: int, 3
T : int, 5
weight_entropy: float, 1.0
weight_diversity:float,1.0

Run example:

python AugTest.py 0 1.0 0.1 0.1 0.1 0.1 0.1 0.1 3 5 1.0 1.0 >>output_GTSRB.txt