# -*- coding: utf-8 -*-
"""MMPINM multi-frequency 2D standard implementation: training entry.

Reproduces Section 4.5 of "A practical PINN framework for multi-scale problems
with multi-magnitude loss terms" (arXiv:2308.06672), MMPINN-MFF method.

  PDE  : u_t = u_xx + f(x,t), x in (-1,1), t in (0,1]
  I.C. : u(x,0) = 0
  B.C. : u(+-1,t) = sin(2*pi*t)
  Exact: u(x,t) = sin(20*pi*t/(1+9x^2))

Training: Adam 2000 iters (lr=1e-3) -> 3-segment L-BFGS (loss^(1/3),^(1/2),^1),
matching the multi-magnitude regularization strategy (m=n=3).

Usage:
  python train.py --epochs 2000 --device cpu --outdir out
  python train.py --lbfgs 100000 --device cuda:0 ...
  python train.py --smoke   # tiny run for interface smoke test
"""
import argparse
import os

import numpy as np
import scipy.io as sio
import torch
import yaml


def lhs(n, samples):
    """pyDOE.lhs equivalent: latin hypercube sampling, (samples, n) in [0,1]."""
    H = np.zeros((samples, n))
    for j in range(n):
        u = np.random.rand(samples)
        P = np.random.permutation(samples)
        H[:, j] = (P + u) / samples
    return H


class MMPINN2D:
    def __init__(self, device, seed=1236, layers=(300, 300, 300, 300, 1), smoke=False,
                 sigma=10.0, N0=1200, N_b=1200, N_f=120000, grid_n=1200, lb=None, ub=None):
        self.device = device
        self.lb = np.array([-1.0, 0.0]) if lb is None else np.array(lb, dtype=np.float32)
        self.ub = np.array([1.0, 1.0]) if ub is None else np.array(ub, dtype=np.float32)
        self.smoke = smoke
        self.sigma = sigma

        # exact solution
        self.heatsolution = lambda x, t: np.sin(20 * np.pi * t / (1 + 9 * x ** 2))

        # --- official data generation (under seed) ---
        if smoke:
            N0, N_b, N_f = 80, 80, 500
            grid_n = 100
        x = np.linspace(self.lb[0], self.ub[0], grid_n).flatten()[:, None]
        t = np.linspace(self.lb[1], self.ub[1], grid_n).flatten()[:, None]
        res = np.zeros((len(x), len(t)))
        for i in range(len(x)):
            for j in range(len(t)):
                res[i, j] = self.heatsolution(x[i, 0], t[j, 0])

        idx_x = np.random.choice(x.shape[0], N0, replace=False)
        x0 = x[idx_x, :]
        u0 = res[idx_x, 0:1]
        idx_t = np.random.choice(t.shape[0], N_b, replace=False)
        tb = t[idx_t, :]
        u_lb = res[0, idx_t].flatten()[:, None]
        u_ub = res[-1, idx_t].flatten()[:, None]
        X_f = self.lb + (self.ub - self.lb) * lhs(2, N_f)

        self.x0, self.u0 = x0, u0
        self.tb, self.u_lb, self.u_ub = tb, u_lb, u_ub
        self.X_f = X_f

        # test grid (1200x1200 points)
        self.X_star = self._make_X_star(x, t)
        self.u_star = res.T.flatten()[:, None]

        from model import PINN2D
        self.net = PINN2D(list(layers), self.lb, self.ub, sigma=self.sigma).to(device)

    def _make_X_star(self, x, t):
        X, T = np.meshgrid(x[:, 0], t[:, 0])
        X_star = np.hstack((X.ravel()[:, None], T.ravel()[:, None]))
        return X_star

    # --- residual (analytic source = u_t - u_xx) ---
    def net_uv(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

    def net_f(self, x, t):
        xt = torch.cat([x, t], dim=1).requires_grad_(True)
        x_in, t_in = xt[:, 0:1], xt[:, 1:2]
        u = self.net_uv(x_in, t_in)
        u_t = torch.autograd.grad(u, t_in, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_in, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_in, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        # source = u_t_exact - u_xx_exact (analytic)
        g = 20 * np.pi * t / (1 + 9 * x ** 2)
        u_t_ex = 20 * np.pi * torch.cos(g) / (1 + 9 * x ** 2)
        u_xx_ex = (-129600 * np.pi ** 2 * t ** 2 * x ** 2 * torch.sin(g) / (1 + 9 * x ** 2) ** 4
                   + 12960 * np.pi * t * x ** 2 * torch.cos(g) / (1 + 9 * x ** 2) ** 3
                   - 360 * np.pi * t * torch.cos(g) / (1 + 9 * x ** 2) ** 2)
        source = u_t_ex - u_xx_ex
        return u_t - u_xx - source

    def loss_fn(self, p=1.0 / 3):
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        # initial (t=0)
        u0_pred = self.net_uv(t(self.x0), t(np.zeros_like(self.x0)))
        # boundary x=-1 / x=1
        u_lb_pred = self.net_uv(t(np.full_like(self.tb, -1.0)), t(self.tb))
        u_ub_pred = self.net_uv(t(np.full_like(self.tb, 1.0)), t(self.tb))
        ic = torch.mean((u0_pred - t(self.u0)) ** 2)
        bc = torch.mean((u_ub_pred - t(self.u_ub)) ** 2) + torch.mean((u_lb_pred - t(self.u_lb)) ** 2)
        # residual
        f_pred = self.net_f(t(self.X_f[:, 0:1]), t(self.X_f[:, 1:2]))
        res = torch.mean(f_pred ** 2)
        # MMPINN multi-magnitude loss: (ic+bc)^p + res^p, p=1/3 (m=n=3)
        loss = torch.pow(ic + bc, p) + torch.pow(res, p)
        return loss, ic, bc, res

    def l2_error(self):
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        X_star = self.X_star
        with torch.no_grad():
            u_pred = self.net_uv(t(X_star[:, 0:1]), t(X_star[:, 1:2])).cpu().numpy()
        err = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
        return err, u_pred

    def train_adam(self, epochs, log_interval=200):
        opt = torch.optim.Adam(self.net.parameters(), lr=0.001)
        import time
        start = time.time()
        for it in range(epochs):
            opt.zero_grad()
            loss, ic, bc, res = self.loss_fn()
            loss.backward()
            opt.step()
            if it % log_interval == 0:
                err, _ = self.l2_error()
                el = time.time() - start
                print(f'Adam It: {it}, Loss: {loss.item():.4e}, IC: {ic.item():.4e}, '
                      f'BC: {bc.item():.4e}, Res: {res.item():.4e}, L2: {err:.3e}, Time: {el:.2f}')
                start = time.time()
        return loss.item()

    def train_lbfgs(self, max_iter=100000, log_interval=50, pat_stop=200):
        """L-BFGS full-batch, 3-segment exponents [1/3, 1/2, 1] (multi-level)."""
        exps = [1.0 / 3, 1.0 / 2, 1.0]
        total_evals = 0
        best_l2 = float('inf')
        for seg, p in enumerate(exps):
            print(f'--- L-BFGS segment {seg + 1}/{len(exps)}: loss^(1/{1.0 / p:.1f}) ---')
            opt = torch.optim.LBFGS(self.net.parameters(), max_iter=max_iter,
                                    line_search_fn='strong_wolfe')
            evals = [0]
            prev_loss = [float('inf')]
            stall = [0]

            def closure():
                opt.zero_grad()
                loss, *_ = self.loss_fn(p)
                loss.backward()
                evals[0] += 1
                if evals[0] % log_interval == 0:
                    err, _ = self.l2_error()
                    print(f'LBFGS seg{seg + 1} step: {evals[0]}, Loss: {loss.item():.4e}, L2: {err:.3e}')
                if loss.item() < prev_loss[0] * (1 - 1e-5) or loss.item() == 0:
                    stall[0] = 0
                else:
                    stall[0] += 1
                prev_loss[0] = loss.item()
                return loss

            import time
            start = time.time()
            opt.step(closure)
            el = time.time() - start
            err, _ = self.l2_error()
            best_l2 = min(best_l2, err)
            print(f'L-BFGS segment {seg + 1} done: {evals[0]} evals, {el:.1f}s, L2: {err:.3e}')
            total_evals += evals[0]
            if err < 1e-3:
                print('L2 < 1e-3 achieved, early stop.')
                break
        print(f'L-BFGS total: {total_evals} evals, best L2: {best_l2:.3e}')
        return total_evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--lbfgs', type=int, default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    # load config.yaml (paper parameters), CLI overrides take precedence
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg.get('seed', 1236)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = args.device or cfg.get('device', 'cuda:0')
    if device != 'cpu' and not torch.cuda.is_available():
        device = 'cpu'
    layers = tuple(cfg.get('layers', [300, 300, 300, 300, 1]))
    sigma = cfg.get('sigma', 10.0)
    N0 = cfg.get('N0', 1200)
    N_b = cfg.get('N_b', 1200)
    N_f = cfg.get('N_f', 120000)
    grid_n = cfg.get('grid_n', 1200)
    lb = cfg.get('lb')
    ub = cfg.get('ub')
    epochs = args.epochs if args.epochs is not None else cfg.get('adam_epochs', 2000)
    lbfgs = args.lbfgs if args.lbfgs is not None else cfg.get('lbfgs_max_iter', 100000)
    outdir = args.outdir or cfg.get('outdir', 'out_2d')

    m = MMPINN2D(device, seed=seed, layers=layers, smoke=args.smoke,
                 sigma=sigma, N0=N0, N_b=N_b, N_f=N_f, grid_n=grid_n, lb=lb, ub=ub)
    m.train_adam(epochs)
    if lbfgs > 0:
        m.train_lbfgs(lbfgs)

    err, u_pred = m.l2_error()
    print(f'FINAL Relative L2 error_u: {err:.2e}')
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        torch.save({'model': m.net.state_dict()},
                   os.path.join(outdir, 'model_2d.pth'))
        sio.savemat(os.path.join(outdir, '1.mat'), {'u': u_pred})
        print('saved ->', outdir)


if __name__ == '__main__':
    main()
