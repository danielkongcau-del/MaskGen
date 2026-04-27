from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ManualARTransformerConfig:
    vocab_size: int
    block_size: int = 4096
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ManualARTransformerConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.dropout = float(config.dropout)
        self.block_size = int(config.block_size)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        q = q.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present = (k, v) if use_cache else None
        key_len = k.size(2)
        past_len = key_len - seq_len
        if self.flash and past_kv is None:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        elif self.flash and seq_len == 1:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if past_kv is None and key_len <= self.block_size:
                mask = self.causal_mask[:, :, :seq_len, :key_len]
            else:
                query_positions = torch.arange(past_len, past_len + seq_len, device=x.device)
                key_positions = torch.arange(0, key_len, device=x.device)
                mask = (key_positions[None, :] <= query_positions[:, None]).view(1, 1, seq_len, key_len)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y)), present


class MLP(nn.Module):
    def __init__(self, config: ManualARTransformerConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: ManualARTransformerConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        attn_output, present = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return (x, present) if use_cache else x


class ManualARTransformer(nn.Module):
    supports_kv_cache = True

    def __init__(self, config: ManualARTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            count -= self.transformer["wpe"].weight.numel()
        return int(count)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> Dict[str, object]:
        _, seq_len = input_ids.shape
        past_len = 0 if past_kv is None else int(past_kv[0][0].size(2))
        if past_len + seq_len > self.config.block_size:
            raise ValueError(f"sequence length {past_len + seq_len} exceeds block_size {self.config.block_size}")
        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device)
        x = self.transformer["wte"](input_ids) + self.transformer["wpe"](positions)[None, :, :]
        x = self.transformer["drop"](x)
        presents = [] if use_cache else None
        for block_index, block in enumerate(self.transformer["h"]):
            layer_past = None if past_kv is None else past_kv[block_index]
            if use_cache:
                x, present = block(x, past_kv=layer_past, use_cache=True)
                presents.append(present)
            else:
                x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            effective_labels = labels
            if attention_mask is not None:
                effective_labels = labels.masked_fill(~attention_mask.bool(), -100)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), effective_labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "past_kv": None if presents is None else tuple(presents)}

    def _sample_next_id(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / float(temperature)
        if top_k is not None and int(top_k) > 0:
            values, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def _generate_without_cache(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int | None,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        for _ in range(int(max_new_tokens)):
            context = input_ids[:, -self.config.block_size :]
            logits = self(context)["logits"][:, -1, :]
            next_id = self._sample_next_id(logits, temperature=temperature, top_k=top_k)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if eos_id is not None and bool((next_id == int(eos_id)).all()):
                break
        return input_ids

    @torch.no_grad()
    def _generate_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int | None,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        outputs = self(input_ids, use_cache=True)
        past_kv = outputs["past_kv"]
        logits = outputs["logits"][:, -1, :]
        for _ in range(int(max_new_tokens)):
            next_id = self._sample_next_id(logits, temperature=temperature, top_k=top_k)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if eos_id is not None and bool((next_id == int(eos_id)).all()):
                break
            outputs = self(next_id, past_kv=past_kv, use_cache=True)
            past_kv = outputs["past_kv"]
            logits = outputs["logits"][:, -1, :]
        return input_ids

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if bool(use_cache) and input_ids.size(1) + int(max_new_tokens) <= self.config.block_size:
            return self._generate_with_cache(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                eos_id=eos_id,
                temperature=float(temperature),
                top_k=top_k,
            )
        return self._generate_without_cache(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            eos_id=eos_id,
            temperature=float(temperature),
            top_k=top_k,
        )

    def configure_optimizers(
        self,
        *,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        param_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}
        decay_params = [param for _, param in param_dict.items() if param.dim() >= 2]
        nodecay_params = [param for _, param in param_dict.items() if param.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = bool(fused_available and device_type == "cuda")
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(
            optim_groups,
            lr=float(learning_rate),
            betas=betas,
            **extra_args,
        )
