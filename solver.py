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

        self.num_updates = 1000
        self.lr = 1e-3 
        self.prev_final_loss = float('inf')  # 이전 loss 저장

        self.lambda_reg = 1e-4 
        
    def set_init(self):
        self.prev_final_loss = float('inf')
    
    def set_input(self, measurement, mask):
        self.measurement = measurement
        self.mask = mask # The value set to 1 in unmasked region

    @torch.no_grad()
    def hard_consistency(self, pred_x0, model, index):
        x0 = model.decode_first_stage(pred_x0)  # D_phi(z_0)
        final_hat_x0 = self.mask * self.measurement + (1. - self.mask) * x0
        if index == 0:
            return final_hat_x0
            
        with torch.no_grad():
            final_hat_z0 = model.get_first_stage_encoding(model.encode_first_stage(final_hat_x0))
            #final_hat_z0 = model.first_stage_model.encode(x0).mode() * 0.18215
        return final_hat_z0

    def cal_grad(self, pred_x0, model, index):
        with torch.no_grad():
            x0 = model.decode_first_stage(pred_x0)  # D_phi(z_0)

        hat_x0 = nn.Parameter(x0.clone())
        optimizer = torch.optim.AdamW([hat_x0], lr=self.lr, weight_decay=0.0) # weight decay로 인해 masked region의 값이 grad가 0이어도 바뀐다, 또한, parameter가 아닌 input을 optimization하는 경우 weight deacy는 오히려 혼동을 준다.

        lambda_reg = self.lambda_reg  # hyperparameter for regularization

        for i in range(self.num_updates):
            optimizer.zero_grad()

            known_hat_x0 = hat_x0 * self.mask
            mc_loss = ((known_hat_x0 - self.measurement)**2).sum() / self.mask.sum()
            reg_loss = lambda_reg * ((hat_x0 - x0.detach())**2 * self.mask).sum() / self.mask.sum()
            loss = mc_loss + reg_loss
            if loss.item() > self.prev_final_loss:
                # print(f"[중단] 현재 초기 loss {loss.item():.6f} > 이전 loss {self.prev_final_loss:.6f}")
                break

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                hat_x0.data = hat_x0 * self.mask + x0.detach() * (1 - self.mask)
            self.prev_final_loss = loss.item()
            # if (i+1) % 100 == 0 or i==0: 
            #    print(f"Update {i+1}/{self.num_updates}, MC: {mc_loss.item()}, Reg: {reg_loss.item()}, Loss: {loss.item()}")
            
        self.set_init()
        final_hat_x0 = hat_x0 * self.mask + x0.detach() * (1 - self.mask)
        final_hat_x0 = final_hat_x0.detach()
        if index == 0:
            return final_hat_x0
            
        with torch.no_grad():
            final_hat_z0 = model.get_first_stage_encoding(model.encode_first_stage(final_hat_x0))
            #final_hat_z0 = model.first_stage_model.encode(x0).mode() * 0.18215
        return final_hat_z0