"""Stage-2 deviation-prototype aggregation for per-part facial features."""

"""
----Readme----
核心概念：

anchor：每个样本、每个部位在时间维的“参考特征”（你代码里是 masked mean）

raw deviation：当前帧相对 anchor 的“偏差表征”（由一个 MLP 生成，不是简单相减）

prototypes：每个部位一组可学习的原型向量（P 个）

attention over prototypes：每个时刻把 raw deviation 映射成对 prototypes 的权重分布

recon deviation：用 prototypes 加权和重构出来的 deviation（相当于把 deviation 投影到原型字典上）

"""



import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .stage2_config import Stage2Config


def _make_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_part_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, use_dropout: bool, dropout_p: float, activation: str
) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        _make_activation(activation),
    ]
    if use_dropout:
        layers.append(nn.Dropout(dropout_p))
    layers.extend(
        [
            nn.Linear(hidden_dim, hidden_dim),
            _make_activation(activation),
        ]
    )
    if use_dropout:
        layers.append(nn.Dropout(dropout_p))
    layers.extend(
        [
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        ]
    )
    return nn.Sequential(*layers)


class Stage2DeviationPrototypeModule(nn.Module):
    """Compute per-frame deviation features with part-specific prototypes."""

    def __init__(self, config: Stage2Config) -> None:
        super().__init__()
        self.config = config
        self.part_names: List[str] = list(config.part_names)
        k = config.num_parts
        d_in = config.d_in
        c = config.latent_dim

        self.proj_layers = nn.ModuleList([nn.Linear(d_in, c) for _ in range(k)])
        self.mlp_layers = nn.ModuleList(
            [
                _build_part_mlp(
                    input_dim=2 * c,
                    hidden_dim=config.hidden_dim,
                    output_dim=c,
                    use_dropout=config.use_dropout,
                    dropout_p=config.dropout_p,
                    activation=config.activation,
                )
                for _ in range(k)
            ]
        )
        self.prototypes = nn.ParameterList(
            [
                nn.Parameter(torch.randn(config.num_prototypes, c) * 0.02)
                for _ in range(k)
            ]
        )
        self.scale = 1.0 / math.sqrt(float(c))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(config.logit_scale_init)))
        self.logit_scale_max = float(config.logit_scale_max)
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(-2.0)) for _ in range(k)])

    def forward(
        self, part_feats: torch.Tensor, present: torch.Tensor
    ) -> Dict[str, torch.Tensor | List[str] | Dict[str, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            B:batch size  T:时间长度（clip内帧数） K:部位数  P:情感原型的个数
            part_feats: (B, T, K, D_in)
            present: (B, T, K) bool mask
        """
        if part_feats.dim() != 4:
            raise ValueError(f"part_feats must be 4D (B,T,K,D); got {part_feats.shape}")
        if present.shape[:3] != part_feats.shape[:3]:
            raise ValueError("present mask must match first three dims of part_feats.")
        b, t, k, d = part_feats.shape
        if k != self.config.num_parts:
            raise ValueError(f"Expected {self.config.num_parts} parts, but got {k}.")
        if d != self.config.d_in:
            raise ValueError(f"Expected d_in={self.config.d_in}, but got {d}.")

        mask = present.float().unsqueeze(-1)  # (B,T,K,1)
        denom = mask.sum(dim=1, keepdim=True)  # (B,1,K,1)
        anchor = (part_feats * mask).sum(dim=1, keepdim=True) / (denom + self.config.eps)  # (B,1,K,D)
        part_has_any_valid = (denom.squeeze(-1).squeeze(1) > 0)  # (B,K) bool

        raw_dev_list: List[torch.Tensor] = []
        raw_dev_anchor_list: List[torch.Tensor] = []
        recon_list: List[torch.Tensor] = []
        recon_scores_list: List[torch.Tensor] = []
        recon_probs_list: List[torch.Tensor] = []
        anchor_scores_list: List[torch.Tensor] = []
        anchor_probs_list: List[torch.Tensor] = []
        top1_list: List[torch.Tensor] = []
        top2_list: List[torch.Tensor] = []
        anchor_top1_list: List[torch.Tensor] = []
        anchor_feat_list: List[torch.Tensor] = []
        mixed_list: List[torch.Tensor] = []

        for idx in range(k):
            proj = self.proj_layers[idx]
            mlp = self.mlp_layers[idx]
            proto = self.prototypes[idx]

            z_k = part_feats[:, :, idx, :]  # (B,T,D)
            anchor_k = anchor[:, :, idx, :]  # (B,1,D)
            anchor_feat_list.append(anchor_k)

            z_proj = proj(z_k)  # (B,T,C)
            anchor_proj = proj(anchor_k)  # (B,1,C)

            anchor_broadcast = anchor_proj.expand(-1, t, -1)
            mlp_in = torch.cat([z_proj, anchor_broadcast], dim=-1)  # (B,T,2C)
            d_k = mlp(mlp_in)  # (B,T,C)

            mlp_anchor_in = torch.cat([anchor_proj, anchor_proj], dim=-1)  # (B,1,2C)
            d_anchor_k = mlp(mlp_anchor_in)  # (B,1,C)

            scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
            scores = torch.einsum("btc,mc->btm", d_k, proto) * self.scale * scale  # (B,T,M)
            probs = torch.softmax(scores, dim=-1)
            recon = torch.einsum("btm,mc->btc", probs, proto)

            anchor_scores = torch.einsum("bqc,mc->bqm", d_anchor_k, proto) * self.scale * scale
            anchor_probs = torch.softmax(anchor_scores, dim=-1)

            topk_val = min(2, proto.shape[0])
            topk_scores = torch.topk(scores, k=topk_val, dim=-1)
            top1_idx = topk_scores.indices[..., 0]
            if topk_val >= 2:
                top2_idx = topk_scores.indices[..., 1]
            else:
                top2_idx = torch.zeros_like(top1_idx)
            anchor_top1 = torch.argmax(anchor_scores, dim=-1)

            raw_dev_list.append(d_k)
            raw_dev_anchor_list.append(d_anchor_k)
            recon_list.append(recon)
            recon_scores_list.append(scores)
            recon_probs_list.append(probs)
            anchor_scores_list.append(anchor_scores)
            anchor_probs_list.append(anchor_probs)
            top1_list.append(top1_idx)
            top2_list.append(top2_idx)
            anchor_top1_list.append(anchor_top1)
            mask_k = present[:, :, idx].unsqueeze(-1).float()
            raw_masked_k = d_k * mask_k
            recon_masked_k = recon * mask_k
            g = torch.sigmoid(self.gates[idx])
            mixed_k = (1.0 - g) * raw_masked_k + g * recon_masked_k
            mixed_list.append(mixed_k)

        raw_dev = torch.stack(raw_dev_list, dim=2)  # (B,T,K,C)
        raw_dev_anchor = torch.stack(raw_dev_anchor_list, dim=2)  # (B,1,K,C)
        recon = torch.stack(recon_list, dim=2)  # (B,T,K,C)
        recon_scores = torch.stack(recon_scores_list, dim=2)  # (B,T,K,M)
        recon_probs = torch.stack(recon_probs_list, dim=2)  # (B,T,K,M)
        anchor_scores_all = torch.stack(anchor_scores_list, dim=2)  # (B,1,K,M)
        anchor_probs_all = torch.stack(anchor_probs_list, dim=2)  # (B,1,K,M)
        top1 = torch.stack(top1_list, dim=2)  # (B,T,K)
        top2 = torch.stack(top2_list, dim=2)  # (B,T,K)
        anchor_top1 = torch.stack(anchor_top1_list, dim=2)  # (B,1,K)
        anchor_feat = torch.stack(anchor_feat_list, dim=2)  # (B,1,K,D)
        mixed = torch.stack(mixed_list, dim=2)  # (B,T,K,C)

        mask_float = mask  # (B,T,K,1)
        raw_dev_masked = raw_dev * mask_float
        recon_masked = recon * mask_float

        fused_output = torch.cat([mixed[:, :, i, :] for i in range(k)], dim=-1)  # (B,T,K*C)

        part_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for idx, name in enumerate(self.part_names):
            part_outputs[name] = {

                "raw_deviation": raw_dev[:, :, idx, :],                           #每个部位、每个时间步的“偏差向量”
                "raw_deviation_masked": raw_dev_masked[:, :, idx, :],             #raw_deviation * valid_mask，部位缺失/无效帧不参与后续重构/损失
                "anchor_deviation": raw_dev_anchor[:, :, idx, :],                 #anchor 自己喂进同一套 deviation-MLP 得到的“和自己的偏差”，用于l_dev计算
                "prototypes": self.prototypes[idx],                               #部位情感原型
                "attention_probs": recon_probs[:, :, idx, :],                     #raw_deviation 与每个 prototype 的相似度打分
                "attention_scores": recon_scores[:, :, idx, :],                   #softmax(attention_scores)，每个时间步对 prototypes 的权重分布
                "recon_deviation": recon[:, :, idx, :],                           #用 attention_probs 对 prototypes 加权和，重构得到的 deviation（部位加权偏差）
                "recon_deviation_masked": recon_masked[:, :, idx, :],             #recon_deviation * valid_mask
                "mixed_deviation": mixed[:, :, idx, :],                           #
                "anchor_attention_probs": anchor_probs_all[:, :, idx, :],         #
                "anchor_attention_scores": anchor_scores_all[:, :, idx, :],
                "top1_idx": top1[:, :, idx],                                      #每个时间步最大概率，以及第二大概率的prototype下标
                "top2_idx": top2[:, :, idx],
                "anchor_top1_idx": anchor_top1[:, :, idx],                        #anchor的top1 prototype 下标
                "valid_mask": present[:, :, idx],                                 #该部位该时间步是否有效
                "part_has_any_valid": part_has_any_valid[:, idx],                 #该部位是否至少有一个有效帧
                "anchor_feat": anchor_feat[:, :, idx, :],
                "gate": torch.sigmoid(self.gates[idx]),                           #当前门控g值，控制SSDL部分起多大作用
            }

        return {
            "fused_output": fused_output,    #“拼装”了各个部位的特征的全局特征
            "present": present,              #来自stage1的part有效性bool掩码。(B,T,K)
            "part_names": self.part_names,   #部位名称列表，长度K
            "part_outputs": part_outputs,  
        
        }


