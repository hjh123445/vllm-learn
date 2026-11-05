"""torchrun --standalone --nproc-per-node=4 v9/tests/test_ep_token_ownership.py.

DP2 x TP2 => EP4; unequal/empty DP inputs and exact route-count oracle.
"""
import os
from test_coupled_moe import worker


class Output:
    def put(self, result):
        if result[0] == 0:
            print({'passed': True, 'scope': 'DP2 x TP2 -> EP4', 'cases': result[1]}, flush=True)


def main():
    os.environ['PP_TEST_BACKEND'] = 'nccl'
    worker(int(os.environ['RANK']), 2, 2, None, Output())


if __name__ == '__main__':
    main()
