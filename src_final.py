import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score as ap_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, precision_recall_curve, auc
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
from tqdm import tqdm


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class EarlyStopper:

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -float("inf")
        self.best_classifier_params = None
        self.best_gate_params = None

    def early_stop(self, validation_score, classifier, gate_net=None):
        if validation_score > self.best_score + self.min_delta:
            self.best_score = validation_score
            self.counter = 0
            self.best_classifier_params = copy.deepcopy(classifier.state_dict())
            if gate_net is not None:
                self.best_gate_params = copy.deepcopy(gate_net.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True, self.best_classifier_params, self.best_gate_params

        return False, self.best_classifier_params, self.best_gate_params


class GeometryGate(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=16, out_mode="fuse_weight"):
        super().__init__()
        self.out_mode = out_mode
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x_euc, x_hyp, return_weights=False):
        x_cat = torch.cat([x_euc, x_hyp], dim=-1)  # [B, 8]
        logits = self.net(x_cat)  # [B, 2]
        weights = torch.softmax(logits, dim=-1)  # [B, 2]

        w_euc = weights[:, 0:1]
        w_hyp = weights[:, 1:2]

        x_fuse = w_euc * x_euc + w_hyp * x_hyp

        if self.out_mode == "fuse":
            out = x_fuse
        elif self.out_mode == "fuse_weight":
            out = torch.cat([x_fuse, weights], dim=-1)  # [B, 6]
        else:
            raise ValueError(f"Unknown out_mode: {self.out_mode}")

        if return_weights:
            return out, weights
        else:
            return out


class MUSE_representation_learning():

    def __init__(self, datasets, device, labels, labels_pos_weights):

        self.datasets = datasets
        self.device = device
        self.labels = labels
        self.labels_pos_weights = labels_pos_weights
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def fit(self, IDXs):

        self.optimizer.zero_grad()

        D_loader = DataLoader(self.datasets, batch_size=len(IDXs), sampler=SubsetSampler(IDXs))
        D = next(iter(D_loader)).to(self.device)
        new_edge_index, edge_id = dropout_edge(D.edge_index, force_undirected=True, p=0.5)
        D.edge_index = new_edge_index

        X_copy = copy.deepcopy(D.x)

        Z = self.model(D)  # Node embeddings
        Z_X = self.feature_head(Z)
        Z_E = self.edge_head(Z)
        L_X = 1 - self.cos(X_copy, Z_X)  # Feature reconstruction via cosine similarity
        curL = 0.0

        TL1 = 0.0
        TL2 = 0.0

        for b_id, idx in zip(IDXs, range(D.ptr.shape[0] - 1)):
            start_indptr = D.ptr[idx]
            end_indptr = D.ptr[idx + 1]
            curZ = Z_E[start_indptr: end_indptr, :]
            A_tilde = torch.matmul(curZ, curZ.T).flatten()

            pos_weight = self.labels_pos_weights[b_id]
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

            L1 = 0.5 * criterion(A_tilde, self.labels[b_id])
            L2 = 0.5 * torch.mean(L_X[start_indptr: end_indptr])

            curL += L1
            curL += L2

            TL1 += L1
            TL2 += L2

        curL /= len(IDXs)
        curL.backward()
        self.optimizer.step()

        TL1 = TL1.detach().cpu().item()
        TL2 = TL2.detach().cpu().item()

        return TL1, TL2

    def train_euc(self, model, feature_head, edge_head, train_idxs, lr=1e-3, weight_decay=1e-6, epochs=200,
                  saving_interval=20, batch_size=50, return_loss=True, fixed_epochs=False, seed=0, pth_path=None):

        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)

        parameters = []
        self.model = model
        self.feature_head = feature_head
        self.edge_head = edge_head
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.feature_head.parameters()) + list(self.edge_head.parameters()),
            lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0)

        train_graphs = np.array(train_idxs)
        total_loss = [[], []]

        for ep in tqdm(range(epochs)):

            np.random.shuffle(train_graphs)
            TL1 = 0
            TL2 = 0

            for idx in range(0, train_graphs.shape[0], batch_size):
                self.model.train()
                self.feature_head.train()
                self.edge_head.train()

                curB = train_graphs[idx: idx + batch_size]
                loss1, loss2 = self.fit(curB)
                TL1 += loss1
                TL2 += loss2

            if return_loss:
                total_loss[0].append(TL1 / len(train_idxs))
                total_loss[1].append(TL2 / len(train_idxs))

            self.scheduler.step()

            if int(ep + 1) % saving_interval == 0:
                parameters.append([copy.deepcopy(self.model.state_dict()),
                                   copy.deepcopy(self.feature_head.state_dict()),
                                   copy.deepcopy(self.edge_head.state_dict())])

        torch.save(self.model.state_dict(), f'{pth_path}/euc_model_state_gating_{seed}.pth')
        torch.save(self.feature_head.state_dict(), f'{pth_path}/euc_feature_head_state_gating_{seed}.pth')
        torch.save(self.edge_head.state_dict(), f'{pth_path}/euc_edge_head_state_gating_{seed}.pth')
        if return_loss:
            return total_loss, parameters
        else:
            return parameters

    def train_hyp(self, model, feature_head, edge_head, train_idxs, lr=1e-3, weight_decay=1e-6, epochs=200,
                  saving_interval=20, batch_size=50, return_loss=True, fixed_epochs=False, seed=0, pth_path=None):

        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)

        parameters = []
        self.model = model
        self.feature_head = feature_head
        self.edge_head = edge_head
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.feature_head.parameters()) + list(self.edge_head.parameters()),
            lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0)

        train_graphs = np.array(train_idxs)
        total_loss = [[], []]

        for ep in tqdm(range(epochs)):

            np.random.shuffle(train_graphs)
            TL1 = 0
            TL2 = 0

            for idx in range(0, train_graphs.shape[0], batch_size):
                self.model.train()
                self.feature_head.train()
                self.edge_head.train()

                curB = train_graphs[idx: idx + batch_size]
                loss1, loss2 = self.fit(curB)
                TL1 += loss1
                TL2 += loss2

            if return_loss:
                total_loss[0].append(TL1 / len(train_idxs))
                total_loss[1].append(TL2 / len(train_idxs))

            self.scheduler.step()

            if int(ep + 1) % saving_interval == 0:
                parameters.append([copy.deepcopy(self.model.state_dict()),
                                   copy.deepcopy(self.feature_head.state_dict()),
                                   copy.deepcopy(self.edge_head.state_dict())])

        torch.save(self.model.state_dict(), f'{pth_path}/hyp_model_state_gating_{seed}.pth')
        torch.save(self.feature_head.state_dict(), f'{pth_path}/hyp_feature_head_state_gating_{seed}.pth')
        torch.save(self.edge_head.state_dict(), f'{pth_path}/hyp_edge_head_state_gating_{seed}.pth')
        if return_loss:
            return total_loss, parameters
        else:
            return parameters


class MUSE_oneclass_classification:

    def __init__(self,
                 model1, feature_encoder1, edge_encoder1,
                 model2, feature_encoder2, edge_encoder2,
                 gate_net,
                 datasets, device, labels, pos_weights,
                 B_size=30, with_feature=True):

        self.datasets = datasets
        self.device = device
        self.B_size = B_size
        self.with_feature = with_feature

        self.model1 = model1
        self.model2 = model2
        self.feature_encoder1 = feature_encoder1
        self.feature_encoder2 = feature_encoder2
        self.edge_encoder1 = edge_encoder1
        self.edge_encoder2 = edge_encoder2

        self.gate_net = gate_net.to(device)

        self.adj_labels = labels
        self.labels_pos_weights = torch.tensor(pos_weights, dtype=torch.float32, device=device)

        self.TX = []
        self.TX_euc = []  # [N,4]
        self.TX_hyp = []  # [N,4]

        self.euc_mean = []
        self.euc_std = []
        self.labels = labels
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    class GeometryGate(nn.Module):
        def __init__(self, in_dim=8, hidden_dim=16, out_mode="fuse_weight"):
            super().__init__()
            self.out_mode = out_mode
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )

        def forward(self, x_euc, x_hyp, return_weights=False):
            x_cat = torch.cat([x_euc, x_hyp], dim=-1)  # [B, 8]
            logits = self.net(x_cat)  # [B, 2]
            weights = torch.softmax(logits, dim=-1)  # [B, 2]

            w_euc = weights[:, 0:1]
            w_hyp = weights[:, 1:2]

            x_fuse = w_euc * x_euc + w_hyp * x_hyp

            if self.out_mode == "fuse":
                out = x_fuse
            elif self.out_mode == "fuse_weight":
                out = torch.cat([x_fuse, weights], dim=-1)  # [B, 6]
            else:
                raise ValueError(f"Unknown out_mode: {self.out_mode}")

            if return_weights:
                return out, weights
            else:
                return out

    def obtain_error_representations(self, parameter, parameter_hyp, train_idxs=None):
        self.TX = []
        self.TX_euc = []
        self.TX_hyp = []
        self.euc_mean = []
        self.euc_std = []
        self.labels = []

        IDXs = list(np.arange(len(self.datasets)))
        B_size = self.B_size
        nG = len(IDXs)

        with torch.no_grad():
            self.model1.load_state_dict(parameter[0])
            self.model2.load_state_dict(parameter_hyp[0])

            self.feature_encoder1.load_state_dict(parameter[1])
            self.feature_encoder2.load_state_dict(parameter_hyp[1])

            self.edge_encoder1.load_state_dict(parameter[2])
            self.edge_encoder2.load_state_dict(parameter_hyp[2])

            self.model1.eval()
            self.model2.eval()
            self.feature_encoder1.eval()
            self.feature_encoder2.eval()
            self.edge_encoder1.eval()
            self.edge_encoder2.eval()

            criterion1 = torch.nn.BCEWithLogitsLoss(reduction='none')
            criterion2 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

            for idx in range(0, nG, B_size):
                curB = IDXs[idx: idx + B_size]

                D_loader = DataLoader(
                    self.datasets,
                    batch_size=len(curB),
                    sampler=SubsetSampler(curB)
                )
                D = next(iter(D_loader)).to(self.device)

                GOAL = copy.deepcopy(D.x)

                # Euclidean
                Z = self.model1(D).detach()
                Z1 = self.feature_encoder1(Z)
                Z2 = self.edge_encoder1(Z)
                X_loss = (1 - criterion2(Z1, GOAL)).cpu()

                # Hyperbolic
                Z_hyp = self.model2(D).detach()
                Z1_hyp = self.feature_encoder2(Z_hyp)
                Z2_hyp = self.edge_encoder2(Z_hyp)
                X_loss_hyp = (1 - criterion2(Z1_hyp, GOAL)).cpu()

                for b_id, local_idx in zip(curB, range(D.ptr.shape[0] - 1)):
                    start_indptr = D.ptr[local_idx]
                    end_indptr = D.ptr[local_idx + 1]

                    curZ = Z2[start_indptr:end_indptr, :]
                    curZ_hyp = Z2_hyp[start_indptr:end_indptr, :]

                    A_torch = torch.matmul(curZ, curZ.T).flatten()
                    A_torch_hyp = torch.matmul(curZ_hyp, curZ_hyp.T).flatten()

                    TL = criterion1(A_torch, self.adj_labels[b_id]).cpu().flatten()
                    TL_hyp = criterion1(A_torch_hyp, self.adj_labels[b_id]).cpu().flatten()

                    avg_L1 = TL.mean().item()
                    std_L1 = TL.std().item()

                    avg_L1_hyp = TL_hyp.mean().item()
                    std_L1_hyp = TL_hyp.std().item()

                    avg_X = torch.mean(X_loss[start_indptr:end_indptr]).item()
                    std_X = torch.std(X_loss[start_indptr:end_indptr]).item()

                    avg_X_hyp = torch.mean(X_loss_hyp[start_indptr:end_indptr]).item()
                    std_X_hyp = torch.std(X_loss_hyp[start_indptr:end_indptr]).item()

                    # 保存一些兼容旧逻辑的统计
                    self.euc_std.append(0.5 * (std_X + std_L1))
                    self.euc_mean.append(0.5 * (avg_X + avg_L1))
                    self.labels.append(self.datasets[b_id].y.item())

                    # 分开保存欧式和双曲的4维统计量
                    x_euc = torch.tensor(
                        [avg_L1, std_L1, avg_X, std_X],
                        dtype=torch.float32, device=self.device
                    )
                    x_hyp = torch.tensor(
                        [avg_L1_hyp, std_L1_hyp, avg_X_hyp, std_X_hyp],
                        dtype=torch.float32, device=self.device
                    )

                    self.TX_euc.append(x_euc)
                    self.TX_hyp.append(x_hyp)

                del Z, Z1, Z2, Z_hyp, Z1_hyp, Z2_hyp, D, GOAL

        self.TX_euc = torch.stack(self.TX_euc).to(self.device)  # [N,4]
        self.TX_hyp = torch.stack(self.TX_hyp).to(self.device)  # [N,4]

        with torch.no_grad():
            self.gate_net.eval()
            self.TX = self.gate_net(self.TX_euc, self.TX_hyp).detach()

    def get_fused_TX(self):
        return self.gate_net(self.TX_euc, self.TX_hyp)

    def polynomial_pacing_function(self, t, T, K, lambda_param=1, min_frac=0.0):
        value = ((t / T) ** lambda_param) * K
        return int(value)

    def compute_multivariate_normal(self, train_idxs):
        TX_fused = self.get_fused_TX().detach()

        if len(train_idxs) == 0:
            D = TX_fused.shape[1]
            mu = torch.zeros(D, device=TX_fused.device)
            cov = torch.eye(D, device=TX_fused.device)
            return mu, cov

        class_samples = TX_fused[train_idxs]  # [N, D]
        N, D = class_samples.shape

        mu = torch.mean(class_samples, dim=0)

        if N == 1:
            cov = torch.eye(D, device=TX_fused.device) * 1e-3
            return mu, cov

        centered = class_samples - mu.unsqueeze(0)
        cov = (centered.T @ centered) / (N - 1)
        cov += 1e-6 * torch.eye(D, device=TX_fused.device)

        return mu, cov

    def kl_divergence_single_sample(self, x, mu, cov):
        D = x.shape[0]
        x_centered = x - mu

        if torch.isnan(cov).any() or torch.isinf(cov).any():
            print("Warning: cov contains NaN/Inf, replacing with identity")
            cov = torch.eye(D, device=cov.device, dtype=cov.dtype)

        cov = (cov + cov.T) / 2.0

        with torch.no_grad():
            diag_mean = torch.mean(torch.diag(cov)).abs()
            eps = max(1e-3, 1e-2 * diag_mean.item() if torch.is_tensor(diag_mean) else 1e-3)

        cov_reg = cov + eps * torch.eye(D, device=cov.device, dtype=cov.dtype)

        try:
            L = torch.linalg.cholesky(cov_reg)
            y = torch.cholesky_solve(x_centered.unsqueeze(1), L).squeeze(1)
            mahalanobis = torch.sum(y ** 2)
            log_det_cov = 2 * torch.sum(torch.log(torch.diag(L)))

        except Exception:
            try:
                cov_reg = cov + 10 * eps * torch.eye(D, device=cov.device, dtype=cov.dtype)
                L = torch.linalg.cholesky(cov_reg)
                y = torch.cholesky_solve(x_centered.unsqueeze(1), L).squeeze(1)
                mahalanobis = torch.sum(y ** 2)
                log_det_cov = 2 * torch.sum(torch.log(torch.diag(L)))

            except Exception:
                try:
                    U, S, Vh = torch.linalg.svd(cov_reg)
                    S_clamped = torch.clamp(S, min=eps)
                    S_inv = 1.0 / S_clamped
                    mahalanobis = torch.sum((x_centered @ U * S_inv) ** 2)
                    log_det_cov = torch.sum(torch.log(S_clamped))

                except Exception:
                    print("Warning: All matrix decomposition failed, using isotropic Gaussian fallback")
                    mahalanobis = torch.sum(x_centered ** 2) / eps
                    log_det_cov = torch.tensor(D * math.log(eps), device=x.device, dtype=x.dtype)

        kl = 0.5 * (mahalanobis + log_det_cov + D * math.log(2 * math.pi))

        if torch.isnan(kl) or torch.isinf(kl):
            print(f"Warning: KL is NaN/Inf ({kl}), returning large value")
            kl = torch.tensor(1e6, device=x.device, dtype=x.dtype)

        return kl

    def reorder_training_data(self, train_all_idxs, train_idxs):

        TX_fused = self.get_fused_TX().detach()
        mu, cov = self.compute_multivariate_normal(train_idxs)

        train_idxs_set = set(train_idxs)

        normal_pairs = []
        anom_pairs = []

        for idx in train_all_idxs:
            x = TX_fused[idx]
            kl = self.kl_divergence_single_sample(x, mu, cov).item()

            if idx in train_idxs_set:
                normal_pairs.append((kl, idx))
            else:
                anom_pairs.append((kl, idx))

        sorted_normal = sorted(normal_pairs, key=lambda x: x[0])
        sorted_anom = sorted(anom_pairs, key=lambda x: x[0], reverse=True)

        sorted_normal_idxs = [idx for _, idx in sorted_normal]
        sorted_abnormal_idxs = [idx for _, idx in sorted_anom]

        return (sorted_abnormal_idxs + sorted_normal_idxs)[::-1]

    def compute_gate_targets(self, idxs, tau=0.5):

        x_euc = self.TX_euc[idxs]  # [B,4]
        x_hyp = self.TX_hyp[idxs]  # [B,4]

        score_euc = x_euc[:, 0] + x_euc[:, 2]  # [B]
        score_hyp = x_hyp[:, 0] + x_hyp[:, 2]  # [B]

        scores = torch.stack([score_euc / tau, score_hyp / tau], dim=1)  # [B,2]
        targets = torch.softmax(scores, dim=1)

        return targets

    def train_MLP_CL(self, encoder_param, encoder_param_hyp, classifier,
                     train_idxs, train_all_idxs, valid_idxs, valid_labels,
                     test_idxs, test_labels,
                     lr=1e-3, epochs=200, w_decay=1e-5, saving_interval=20,
                     early_stop=20, seed=0, tau=0.8,
                     lambda_gate=0.5, lambda_ent=1e-3):  # 新增两个超参数

        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)

        if len(self.TX_euc) == 0 or len(self.TX_hyp) == 0:
            self.obtain_error_representations(encoder_param, encoder_param_hyp, train_idxs)

        early_stopper = EarlyStopper(patience=early_stop, min_delta=0.0)

        optimizer = torch.optim.Adam(
            list(classifier.parameters()) + list(self.gate_net.parameters()),
            lr=lr, weight_decay=w_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=0.0
        )

        torch.manual_seed(0)

        sorted_train_all_idxs = self.reorder_training_data(
            train_all_idxs=train_all_idxs,
            train_idxs=train_idxs
        )

        K_norm = len(sorted_train_all_idxs)
        T = epochs

        cur_cls_param = copy.deepcopy(classifier.state_dict())
        cur_gate_param = copy.deepcopy(self.gate_net.state_dict())

        for ep in range(epochs):
            num_train = self.polynomial_pacing_function(ep + 1, T, K_norm, lambda_param=1)
            num_train = max(1, num_train)
            cur_train_idxs = sorted_train_all_idxs[:num_train]

            classifier.train()
            self.gate_net.train()
            optimizer.zero_grad()

            TX_fused, gate_weights = self.gate_net(self.TX_euc, self.TX_hyp, return_weights=True)
            cur_input = TX_fused[cur_train_idxs]
            cur_weights = gate_weights[cur_train_idxs]

            curZ = classifier(cur_input)

            L_recon = torch.mean(torch.sqrt(torch.sum((curZ - cur_input) ** 2, dim=1)))

            gate_targets = self.compute_gate_targets(cur_train_idxs, tau=tau)
            L_gate = F.mse_loss(cur_weights, gate_targets)

            L_ent = -torch.mean(torch.sum(cur_weights * torch.log(cur_weights + 1e-8), dim=1))

            loss = L_recon + lambda_gate * L_gate - lambda_ent * L_ent

            loss.backward()
            optimizer.step()
            scheduler.step()

            if int(ep + 1) % saving_interval == 0:
                stds = self.return_variance(idxs=train_idxs, classifier=classifier)

                cur_val = self.evaluate_with_Reconstruction(
                    valid_idxs, valid_labels, classifier, stds
                )

                cur_res, cur_cls_param, cur_gate_param = early_stopper.early_stop(
                    cur_val, classifier, self.gate_net
                )

                if cur_res:
                    break

        if cur_cls_param is not None:
            classifier.load_state_dict(cur_cls_param)
        if cur_gate_param is not None:
            self.gate_net.load_state_dict(cur_gate_param)

        stds = self.return_variance(idxs=train_idxs, classifier=classifier)

        val_auroc_score, val_auprc_score = self.evaluate_with_Reconstruction(
            valid_idxs, valid_labels, classifier, stds, final=True
        )
        test_auroc_score, test_auprc_score = self.evaluate_with_Reconstruction(
            test_idxs, test_labels, classifier, stds, final=True
        )

        return val_auroc_score, val_auprc_score, test_auroc_score, test_auprc_score,

    def evaluate_with_Reconstruction(self, idxs, label, classifier, stds, final=False, anomaly_quantile=0.2):
        with torch.no_grad():
            classifier.eval()
            self.gate_net.eval()

            TX_fused = self.get_fused_TX()
            cur_input = TX_fused[idxs]
            curZ = classifier(cur_input)

            score = (
                torch.sum(((curZ - cur_input) ** 2) / stds, dim=1)
            ).detach().cpu().numpy()
            label_ = 1 - np.array(label)

        if final:
            auroc_score = roc_auc_score(label_, score)
            precision, recall, thresholds = precision_recall_curve(label_, score)
            auprc_score = auc(recall, precision)
            return auroc_score, auprc_score
        else:
            auroc_score = roc_auc_score(label_, score)
            return auroc_score

    def return_variance(self, idxs, classifier):
        with torch.no_grad():
            classifier.eval()
            self.gate_net.eval()

            TX_fused = self.get_fused_TX()
            curZ = classifier(TX_fused[idxs])

            stds = torch.std(curZ, dim=0) + 1e-6

        return stds
