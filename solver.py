import torch
from torch.autograd import grad
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.nn as nn

class LMGrad():
    def __init__(self):
        super().__init__()
        self.measurement = None
        self.mask = None

        self.num_updates = 1000# 1000
        self.lr = 1e-3
        self.prev_final_loss = float('inf')  # 이전 loss 저장

        self.lambda_reg = 1e-5 # 1은 뭔가 blur한 느낌 , 1e-2는 선명함,  1e-5도 error는 없음
        
    def set_init(self):
        self.prev_final_loss = float('inf')
    
    def set_input(self, measurement, mask):
        self.measurement = measurement
        self.mask = mask

    def cal_grad(self, pred_x0, model, index):
        with torch.no_grad():
            x0 = model.decode_first_stage(pred_x0)  # D_phi(z_0)

        hat_x0 = x0.detach().clone().requires_grad_(True)  # 최적화 대상
        optimizer = torch.optim.AdamW([hat_x0], lr=self.lr)

        lambda_reg = self.lambda_reg  # hyperparameter for regularization

        for i in range(self.num_updates):
            optimizer.zero_grad()

            # Measurement consistency (e.g. masked L1)
            known_hat_x0 = hat_x0 * self.mask
            mc_loss = ((known_hat_x0 - self.measurement) ** 2) * self.mask
            mc_loss = mc_loss.sum() / self.mask.sum()

            # Regularization term: \lambda * ||x - x0||^2
            reg_loss = lambda_reg * (((hat_x0 - x0.detach()) ** 2) * self.mask).sum() / self.mask.sum()

            loss = mc_loss + reg_loss
            if loss.item() > self.prev_final_loss:
                # print(f"[중단] 현재 초기 loss {loss.item():.6f} > 이전 loss {self.prev_final_loss:.6f}")
                break
            loss.backward()
            optimizer.step()
            self.prev_final_loss = loss.item()
            # if (i+1) % 100 == 0 or i==0: 
            #     print(f"Update {i+1}/{self.num_updates}, MC: {mc_loss.item()}, Reg: {reg_loss.item()}, TV: {tv_loss.item()}, Loss: {loss.item()}")

        self.set_init()
        final_hat_x0 = hat_x0.detach_()
        if index == 0:
            return final_hat_x0
            
        with torch.no_grad():
            final_hat_z0 = model.get_first_stage_encoding(model.encode_first_stage(final_hat_x0))
            #final_hat_z0 = model.first_stage_model.encode(x0).mode() * 0.18215
        return final_hat_z0