"""
Training loop for NL-MTP HoF experiment.

Implements:
- Observed-world and policy-world two-pass forward
- DR/AIPW losses (unit-level + scalar ψ)
- MDN propensity loss
- Per-environment REx invariance penalty
- LoR locality penalties
- Optional MMP warmup auxiliary
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F


def _mdn_logpdf(mdn_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], a: torch.Tensor) -> torch.Tensor:
    """Compute log p(a) under MDN."""
    pi, mu, log_sigma = mdn_params
    sigma = torch.exp(log_sigma)
    a = a.unsqueeze(-1)
    comp = -0.5 * (((a - mu) / (sigma + 1e-8)) ** 2 + 2 * log_sigma + torch.log(torch.tensor(2 * 3.141592653589793, device=a.device)))
    return torch.logsumexp(torch.log(pi + 1e-8) + comp, dim=-1)


def _per_env_variance(residuals: torch.Tensor, env_idx: torch.Tensor) -> torch.Tensor:
    """Compute variance of per-environment means (REx-style)."""
    unique_envs = env_idx.unique()
    if len(unique_envs) <= 1:
        return torch.tensor(0.0, device=residuals.device)
    
    env_means = []
    for e in unique_envs:
        mask = (env_idx == e)
        if mask.sum() > 0:
            env_means.append(residuals[mask].mean())
    
    if len(env_means) == 0:
        return torch.tensor(0.0, device=residuals.device)
    
    env_means = torch.stack(env_means)
    return torch.var(env_means)


def train_epoch(
    model,
    loaders: Tuple,
    opt,
    sched,
    delta: float,
    epoch: int,
    device: str = "cuda",
    warmup_epochs: int = 5,
    c_offset: float = 400.0,  # Shift HoF to positive range: min(-300) + 400 = 100
    w_max: float = 20.0,
    support_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Train one epoch with DR/AIPW losses, REx, LoR penalties, and optional MMP warmup.
    
    Args:
        model: NL_MTP_Model
        loaders: (train_dl, id_dl, ood_dl)
        opt: optimizer
        sched: LR scheduler
        delta: policy shift (e.g., +14)
        epoch: current epoch number
        device: torch device
        warmup_epochs: epochs to apply MMP auxiliary
        c_offset: offset for log(Y - c)
        w_max: max importance weight
        support_threshold: minimum support gate value
    
    Returns:
        Dict with train_loss and component losses
    """
    model.train()
    train_dl, _, _ = loaders
    
    loss_sum = 0.0
    loss_obs_sum = 0.0
    loss_mdn_sum = 0.0
    loss_dr_func_sum = 0.0
    loss_dr_mean_sum = 0.0
    loss_rex_sum = 0.0
    loss_lor_sum = 0.0
    n_sum = 0
    
    for batch in train_dl:
        x_ctx = batch["x_ctx"].to(device)
        mw = batch["mw"].to(device)
        y = batch["y"].to(device)
        env_idx = batch["env_idx"].to(device)
        
        B = x_ctx.size(0)
        
        # Stabilized log target (shift to positive range)
        y_log = torch.log(torch.clamp(y + c_offset, min=1e-6))
        
        ###############################
        # Observed world (no LoR)
        ###############################
        out_obs = model(env_idx, x_ctx, mw, delta, apply_lor=False)
        m_obs = out_obs["m_obs"]
        mdn_params = out_obs["mdn_params"]
        support_obs = out_obs["support"]
        
        # Propensity densities
        logp_A = _mdn_logpdf(mdn_params, mw)
        logp_A_minus_delta = _mdn_logpdf(mdn_params, mw - delta)
        
        ###############################
        # Policy world (LoR active)
        ###############################
        out_pol = model(env_idx, x_ctx, mw, delta, apply_lor=True)
        m_pol = out_pol["m_obs"]
        support_pol = out_pol["support"]
        alpha = out_pol["alpha"]
        
        ###############################
        # Support gating and weights
        ###############################
        # Support is min of observed and policy
        support = torch.minimum(support_obs, support_pol)
        support_mask = (support >= support_threshold)
        
        # Importance weight w = g(A-δ|X) / g(A|X)
        log_w = torch.clamp(logp_A_minus_delta - logp_A, min=-5.0, max=3.0)
        w = torch.exp(log_w).clamp(0, w_max)
        
        ###############################
        # DR/AIPW pseudo-outcome
        ###############################
        y_tilde = m_pol + (y_log - m_obs) * w
        
        ###############################
        # Losses
        ###############################
        # L_obs: supervised on observed world
        L_obs = F.mse_loss(m_obs, y_log)
        
        # L_mdn: propensity density
        L_mdn = -logp_A.mean()
        
        # L_DR-func: unit-level DR residual
        if support_mask.sum() > 0:
            L_dr_func = F.mse_loss(m_pol[support_mask], y_tilde[support_mask])
        else:
            L_dr_func = torch.tensor(0.0, device=device)
        
        # L_DR-mean: scalar ψ(δ)
        if support_mask.sum() > 0:
            L_dr_mean = (y_tilde[support_mask].mean() - model.psi_scalar) ** 2
        else:
            L_dr_mean = torch.tensor(0.0, device=device)
        
        # L_rex: per-environment variance of policy residual
        policy_residual = (m_pol - y_tilde) ** 2
        L_rex = _per_env_variance(policy_residual, env_idx)
        
        # L_lor: locality penalties (alpha^2 + adapter Frobenius norms)
        L_lor = (alpha ** 2).mean()
        for i in model.lor_layers:
            L_lor = L_lor + 0.01 * (model.adapters_q[str(i)].U.norm() + model.adapters_q[str(i)].V.norm())
            L_lor = L_lor + 0.01 * (model.adapters_k[str(i)].U.norm() + model.adapters_k[str(i)].V.norm())
            L_lor = L_lor + 0.01 * (model.adapters_o[str(i)].U.norm() + model.adapters_o[str(i)].V.norm())
            L_lor = L_lor + 0.01 * (model.adapters_mlp[str(i)].U.norm() + model.adapters_mlp[str(i)].V.norm())
        
        # L_mmp: MMP warmup auxiliary (self-supervised descriptor delta prediction)
        # During warmup, train the model to predict how molecular weight changes affect descriptors
        L_mmp = torch.tensor(0.0, device=device)
        if epoch <= warmup_epochs:
            # Use the same batch but predict descriptor changes under policy
            with torch.no_grad():
                # Compute actual descriptor changes by feeding perturbed molecules
                # For this implementation, we approximate: descriptors should change
                # in directions consistent with weight increase (e.g., more heavy atoms)
                # We use a simple heuristic: expect small, positive changes in size-related descriptors
                target_desc_delta = torch.zeros(8, device=device)
                # Expected deltas for +14 Da (roughly one CH2 group):
                # MolLogP: slight increase, TPSA: minimal, nHBA/nHBD: minimal, 
                # nRotatableBonds: +1, nRings: 0, molecular complexity: slight increase
                target_desc_delta[0] = 0.5   # MolLogP increase
                target_desc_delta[1] = 0.0   # TPSA minimal
                target_desc_delta[2] = 0.0   # nHBA
                target_desc_delta[3] = 0.0   # nHBD
                target_desc_delta[4] = 1.0   # nRotatableBonds
                target_desc_delta[5] = 0.0   # nRings
                target_desc_delta[6] = 2.0   # nHeavyAtoms (one CH2 = 2 heavy)
                target_desc_delta[7] = 0.3   # Complexity increase
            
            # Predict descriptor delta from policy-world head
            pred_desc_delta = out_pol['desc_delta']  # [B, 8]
            L_mmp = F.l1_loss(pred_desc_delta, target_desc_delta.unsqueeze(0).expand_as(pred_desc_delta))
        
        ###############################
        # Total loss (fixed weights from spec)
        ###############################
        L_total = (
            1.0 * L_dr_func +
            0.2 * L_dr_mean +
            1.0 * L_obs +
            0.5 * L_mdn +
            0.2 * L_rex +
            0.1 * L_lor +
            0.2 * L_mmp
        )
        
        ###############################
        # Backprop and step
        ###############################
        opt.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        
        ###############################
        # Accumulate for reporting
        ###############################
        loss_sum += L_total.item() * B
        loss_obs_sum += L_obs.item() * B
        loss_mdn_sum += L_mdn.item() * B
        loss_dr_func_sum += L_dr_func.item() * B
        loss_dr_mean_sum += L_dr_mean.item() * B
        loss_rex_sum += L_rex.item() * B
        loss_lor_sum += L_lor.item() * B
        n_sum += B
    
    return {
        "train_loss": loss_sum / max(1, n_sum),
        "L_obs": loss_obs_sum / max(1, n_sum),
        "L_mdn": loss_mdn_sum / max(1, n_sum),
        "L_dr_func": loss_dr_func_sum / max(1, n_sum),
        "L_dr_mean": loss_dr_mean_sum / max(1, n_sum),
        "L_rex": loss_rex_sum / max(1, n_sum),
        "L_lor": loss_lor_sum / max(1, n_sum),
    }


@torch.no_grad()
def evaluate(
    model,
    loaders: Tuple,
    delta: float,
    device: str = "cuda",
    c_offset: float = 400.0,  # Must match training offset
) -> Dict[str, Any]:
    """
    Evaluate on ID and OOD splits.
    
    Returns:
        Dict with id_rmse, id_mae, ood_rmse, ood_mae, and policy contrast stats.
    """
    model.eval()
    _, id_dl, ood_dl = loaders
    
    def _eval_dl(dl):
        y_true_list = []
        y_pred_list = []
        y_pol_list = []
        alpha_list = []
        
        for batch in dl:
            x_ctx = batch["x_ctx"].to(device)
            mw = batch["mw"].to(device)
            y = batch["y"].to(device)
            env_idx = batch["env_idx"].to(device)
            
            # Observed world
            out_obs = model(env_idx, x_ctx, mw, delta, apply_lor=False)
            m_obs = out_obs["m_obs"]
            y_hat_obs = torch.exp(m_obs) - c_offset
            
            # Policy world
            out_pol = model(env_idx, x_ctx, mw, delta, apply_lor=True)
            m_pol = out_pol["m_obs"]
            y_hat_pol = torch.exp(m_pol) - c_offset
            alpha = out_pol["alpha"]
            
            y_true_list.append(y)
            y_pred_list.append(y_hat_obs)
            y_pol_list.append(y_hat_pol)
            alpha_list.append(alpha)
        
        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        y_pol = torch.cat(y_pol_list)
        alpha = torch.cat(alpha_list)
        
        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        policy_contrast = (y_pol - y_pred).mean().item()
        alpha_mean = alpha.mean().item()
        
        return rmse, mae, policy_contrast, alpha_mean, y_true, y_pred
    
    id_rmse, id_mae, id_contrast, id_alpha, id_y_true, id_y_pred = _eval_dl(id_dl)
    ood_rmse, ood_mae, ood_contrast, ood_alpha, ood_y_true, ood_y_pred = _eval_dl(ood_dl)
    
    return {
        "id_rmse": id_rmse,
        "id_mae": id_mae,
        "id_policy_contrast": id_contrast,
        "id_alpha": id_alpha,
        "ood_rmse": ood_rmse,
        "ood_mae": ood_mae,
        "ood_policy_contrast": ood_contrast,
        "ood_alpha": ood_alpha,
        "id_y_true": id_y_true.cpu(),
        "id_y_pred": id_y_pred.cpu(),
        "ood_y_true": ood_y_true.cpu(),
        "ood_y_pred": ood_y_pred.cpu(),
    }
