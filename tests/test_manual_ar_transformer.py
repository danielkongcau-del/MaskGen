from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from partition_gen.manual_ar_training import build_optimizer, load_checkpoint, save_checkpoint
from partition_gen.models.manual_ar_transformer import ManualARTransformer, ManualARTransformerConfig


class ManualARTransformerTest(unittest.TestCase):
    def make_model(self) -> ManualARTransformer:
        config = ManualARTransformerConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
            bias=False,
        )
        return ManualARTransformer(config)

    def test_forward_shape_and_loss(self) -> None:
        model = self.make_model()
        input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
        labels = torch.randint(0, 32, (2, 8), dtype=torch.long)
        labels[1, 6:] = -100
        attention_mask = torch.ones((2, 8), dtype=torch.bool)
        attention_mask[1, 6:] = False

        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

        self.assertEqual(tuple(outputs["logits"].shape), (2, 8, 32))
        self.assertTrue(torch.isfinite(outputs["loss"]))

    def test_backward_produces_gradients(self) -> None:
        model = self.make_model()
        input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
        labels = torch.randint(0, 32, (2, 8), dtype=torch.long)
        loss = model(input_ids=input_ids, labels=labels)["loss"]
        loss.backward()
        grad_norm = sum(float(param.grad.abs().sum().item()) for param in model.parameters() if param.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_generate_from_bos_respects_max_length(self) -> None:
        model = self.make_model()
        generated = model.generate(
            torch.tensor([[1]], dtype=torch.long),
            max_new_tokens=5,
            eos_id=2,
            temperature=0.0,
        )
        self.assertLessEqual(generated.shape[1], 6)
        self.assertEqual(int(generated[0, 0].item()), 1)

    def test_cached_forward_matches_full_forward_logits(self) -> None:
        model = self.make_model()
        model.eval()
        input_ids = torch.randint(0, 32, (1, 8), dtype=torch.long)
        full_logits = model(input_ids=input_ids)["logits"]
        past_kv = None
        cached_logits = []
        for index in range(input_ids.size(1)):
            outputs = model(input_ids=input_ids[:, index : index + 1], past_kv=past_kv, use_cache=True)
            past_kv = outputs["past_kv"]
            cached_logits.append(outputs["logits"])
        cached = torch.cat(cached_logits, dim=1)
        self.assertTrue(torch.allclose(full_logits, cached, atol=1e-5, rtol=1e-5))

    def test_cached_greedy_generate_matches_uncached_generate(self) -> None:
        model = self.make_model()
        model.eval()
        start = torch.tensor([[1]], dtype=torch.long)
        cached = model.generate(start.clone(), max_new_tokens=5, temperature=0.0, use_cache=True)
        uncached = model.generate(start.clone(), max_new_tokens=5, temperature=0.0, use_cache=False)
        self.assertTrue(torch.equal(cached, uncached))

    def test_checkpoint_save_load(self) -> None:
        model = self.make_model()
        optimizer = build_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            device_type="cpu",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=None,
                iter_num=3,
                best_val_loss=1.25,
                model_config=model.config,
                train_config={"learning_rate": 1e-3, "weight_decay": 0.01, "beta1": 0.9, "beta2": 0.95},
                vocab_path=Path("vocab.json"),
                special_token_ids={"pad": 0, "bos": 1, "eos": 2, "unk": 3},
            )
            checkpoint, loaded, _ = load_checkpoint(path)

        self.assertEqual(checkpoint["iter_num"], 3)
        self.assertEqual(loaded.config.vocab_size, model.config.vocab_size)
        logits = loaded(torch.randint(0, 32, (1, 4), dtype=torch.long))["logits"]
        self.assertEqual(tuple(logits.shape), (1, 4, 32))


if __name__ == "__main__":
    unittest.main()
