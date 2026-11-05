"""Activation capture must snapshot block inputs before in-place residual updates."""
import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.activation_capture import install_activation_capture


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()
        self.v_proj = nn.Identity()
        self.o_proj = nn.Identity()


class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Identity()


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()
        self.mlp = FakeMLP()

    def forward(self, x, residual, positions):
        residual.add_(x)
        return x + 1, residual


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Identity()
        self.layers = nn.ModuleList([FakeBlock()])
        self.norm = nn.Identity()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBase()
        self.lm_head = nn.Identity()


class ActivationCaptureTests(unittest.TestCase):
    def test_block_input_is_copied_before_residual_mutation(self):
        model = FakeModel()
        x = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[10.0, 20.0]])
        x_before, residual_before = x.clone(), residual.clone()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'VLLM_LEARN_ACTIVATIONS_MODULES': 'block',
                'VLLM_LEARN_ACTIVATIONS_CALL_INDEX': '0',
                'VLLM_LEARN_ACTIVATIONS_FULL_ROWS': '1',
                'VLLM_LEARN_ACTIVATIONS_LAYERS': '0',
                'VLLM_LEARN_ACTIVATIONS_WEIGHTS': ''}):
            install_activation_capture(model, directory)
            model.model.layers[0](x, residual, torch.tensor([0]))
            record = torch.load(Path(directory)/'stage0.layer0.block.pt',
                                map_location='cpu', weights_only=True)

        self.assertTrue(torch.equal(record['input'][0], x_before))
        self.assertTrue(torch.equal(record['input'][1], residual_before))
        self.assertTrue(torch.equal(record['output'][1], residual_before + x_before))


if __name__ == '__main__':
    unittest.main()
