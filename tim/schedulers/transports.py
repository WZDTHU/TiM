import torch
from typing import Optional

class Transport:
    def __init__(self, sigma_d, T_max, T_min, enhance_target=False, w_gt=1.0, w_cond=0.0, w_start=0.0, w_end=1.0):
        self.sigma_d = sigma_d
        self.T_max = T_max
        self.T_min = T_min
        self.enhance_target = enhance_target
        self.w_gt = w_gt
        self.w_cond = w_cond
        self.w_start = w_start
        self.w_end = w_end
    
    def sample_t(self, batch_size, dtype, device):
        pass
    def c_noise(self, t: torch.Tensor):
        pass
    def interpolant(self, t: torch.Tensor):
        pass
    def target(self, x_t: torch.Tensor, v_t: torch.Tensor, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor, dF_dv_dt: torch.Tensor, F_t_cond: torch.Tensor, F_t_uncond: torch.Tensor):
        pass
    def from_x_t_to_x_r(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor, F: torch.Tensor):
        pass

class OT_FM(Transport):
    def __init__(self, P_mean=0.0, P_std=1.0, sigma_d=1.0, T_max=1.0, T_min=0.0, enhance_target=False, w_gt=1.0, w_cond=0.0, w_start=0.0, w_end=1.0):
        '''
        Flow-matching with linear path formulation from the paper:
        "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers"
        '''
        self.P_mean = P_mean
        self.P_std = P_std
        super().__init__(sigma_d, T_max, T_min, enhance_target, w_gt, w_cond, w_start, w_end)
        
    def interpolant(self, t: torch.Tensor):
        alpha_t = 1 - t
        sigma_t = t
        d_alpha_t = -1
        d_sigma_t =  1
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_t(self, batch_size, dtype, device):
        rnd_normal = torch.randn((batch_size, ), dtype=dtype, device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        t = sigma / (1 + sigma)     # [0, 1]
        return t

    def c_noise(self, t: torch.Tensor):
        return t
    
    def target(
        self, 
        x_t: torch.Tensor, 
        v_t: torch.Tensor, 
        x: torch.Tensor, 
        z: torch.Tensor, 
        t: torch.Tensor, 
        r: torch.Tensor, 
        dF_dv_dt: torch.Tensor,
        F_t_cond: Optional[torch.Tensor] = 0.0, 
        F_t_uncond: Optional[torch.Tensor] = 0.0, 
        enhance_target = False,
    ):
        if enhance_target:
            w_gt = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_gt, 1.0)
            w_cond = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_cond, 0.0)
            v_t = w_gt * v_t + w_cond * F_t_cond + (1-w_gt-w_cond) * F_t_uncond
        F_target = v_t - (t - r) * dF_dv_dt
        return F_target
    
    def from_x_t_to_x_r(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor, F: torch.Tensor, s_ratio=0.0):
        x_r = x_t - (t - r) * F
        if s_ratio > 0.0:
            z = x_t + (1-t) * F
            epsilon = torch.randn_like(z)
            dt = t-r
            x_r = x_r - s_ratio * z * dt + torch.sqrt(s_ratio*2*t*dt) * epsilon
        return x_r
    



class TrigFlow(Transport):
    def __init__(self, P_mean=-1.0, P_std=1.6, sigma_d=0.5, T_max=1.57, T_min=0.0, enhance_target=False, w_gt=1.0, w_cond=0.0, w_start=0.0, w_end=1.0):
        '''
        TrigFlow formulation from the paper:
        "SIMPLIFYING, STABILIZING & SCALING CONTINUOUS-TIME CONSISTENCY MODELS"
        '''
        self.P_mean = P_mean
        self.P_std = P_std
        super().__init__(sigma_d, T_max, T_min, enhance_target, w_gt, w_cond, w_start, w_end)
        
    def interpolant(self, t: torch.Tensor):
        alpha_t = torch.cos(t)
        sigma_t = torch.sin(t)
        d_alpha_t = -torch.sin(t)
        d_sigma_t =  torch.cos(t)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_t(self, batch_size, dtype, device):
        rnd_normal = torch.randn((batch_size, ), dtype=dtype, device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        t = torch.atan(sigma)       # [0, pi/2]
        return t

    def c_noise(self, t: torch.Tensor):
        return t
    
    def target(
        self, 
        x_t: torch.Tensor, 
        v_t: torch.Tensor, 
        x: torch.Tensor, 
        z: torch.Tensor, 
        t: torch.Tensor, 
        r: torch.Tensor, 
        dF_dv_dt: torch.Tensor,
        F_t_cond: Optional[torch.Tensor] = 0.0, 
        F_t_uncond: Optional[torch.Tensor] = 0.0, 
        enhance_target = False,
    ):
        if enhance_target:
            w_gt = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_gt, 1.0)
            w_cond = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_cond, 0.0)
            v_t = w_gt * v_t + w_cond * F_t_cond + (1-w_gt-w_cond) * F_t_uncond
        F_target = v_t - torch.tan(t - r) * (x_t + dF_dv_dt)
        return F_target
    
    def from_x_t_to_x_r(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor, F: torch.Tensor, s_ratio=0.0):
        x_r = torch.cos(t - r) * x_t - torch.sin(t - r) * F
        return x_r


class EDM(Transport):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_d=0.5, T_max=80.0, T_min=0.01, enhance_target=False, w_gt=1.0, w_cond=0.0, w_start=0.0, w_end=1.0):
        '''
        EDM formulation from the paper: 
        "Elucidating the Design Space of Diffusion-Based Generative Models"
        '''
        self.P_mean = P_mean
        self.P_std = P_std
        super().__init__(sigma_d, T_max, T_min, enhance_target, w_gt, w_cond, w_start, w_end)
        
    def interpolant(self, t: torch.Tensor):
        '''
        The d_alpha_t and d_sigma_t are easy to obtain: 
        # from sympy import *
        # from scipy.stats import *
        # t, sigma_d = symbols('t sigma_d')
        # alpha_t = sigma_d * ((t**2 + sigma_d**2) ** (-0.5))
        # sigma_t = t * ((t**2 + sigma_d**2) ** (-0.5))
        # d_alpha_t = diff(alpha_t, t)
        # d_sigma_t = diff(sigma_t, t)
        # print(d_alpha_t)
        # print(d_sigma_t)
        '''
        sigma_d = self.sigma_d
        alpha_t = 1 / (t**2 + sigma_d**2).sqrt()
        sigma_t = t / (t**2 + sigma_d**2).sqrt()
        d_alpha_t = -t / ((sigma_d ** 2 + t ** 2) ** 1.5)
        d_sigma_t =  (sigma_d ** 2) / ((sigma_d ** 2 + t ** 2) ** 1.5)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_t(self, batch_size, dtype, device):
        rnd_normal = torch.randn((batch_size, ), dtype=dtype, device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        t = sigma       # t > 0
        return t
    
    def c_noise(self, t: torch.Tensor):
        return torch.log(t) / 4
    
    def target(
        self, 
        x_t: torch.Tensor, 
        v_t: torch.Tensor, 
        x: torch.Tensor, 
        z: torch.Tensor, 
        t: torch.Tensor, 
        r: torch.Tensor, 
        dF_dv_dt: torch.Tensor,
        F_t_cond: Optional[torch.Tensor] = 0.0, 
        F_t_uncond: Optional[torch.Tensor] = 0.0, 
        enhance_target = False,
    ):
        sigma_d = self.sigma_d
        alpha_hat_t = t / (sigma_d * (t**2 + sigma_d**2).sqrt())
        sigma_hat_t = - sigma_d / (t**2 + sigma_d**2).sqrt()
        d_alpha_hat_t = -t**2/(sigma_d*(sigma_d**2 + t**2)**(3/2)) + 1/(sigma_d*(sigma_d**2 + t**2).sqrt())
        d_sigma_hat_t = sigma_d * t / ((sigma_d**2 + t**2)**(3/2))
        diffusion_target = alpha_hat_t * x + sigma_hat_t * z
        Bt_dv_dBt = (t - r) * (sigma_d**2 + t**2) * (sigma_d**3 + t**2) / (
            2*t*(r - t)*(sigma_d**2 + t**2) - t*(r - t)*(sigma_d**3 + t**2) + (sigma_d**2 + t**2)*(sigma_d**3 + t**2)
        )
        if enhance_target:
            w_gt = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_gt, 1.0)
            w_cond = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_cond, 0.0)
            diffusion_target = w_gt * diffusion_target + w_cond * F_t_cond + (1-w_gt-w_cond) * F_t_uncond
        F_target = diffusion_target + Bt_dv_dBt * (d_alpha_hat_t*x + d_sigma_hat_t*z -dF_dv_dt)
        return F_target


    def from_x_t_to_x_r(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor, F: torch.Tensor, s_ratio=0.0):
        sigma_d = self.sigma_d
        ratio = (t**2 + sigma_d**2).sqrt() / (r**2 + sigma_d**2).sqrt() / (sigma_d**3 + t**2)
        A_t = (sigma_d**3 + t*r) * ratio
        B_t = (sigma_d**2) * (t-r) * ratio
        x_r = A_t * x_t + B_t * F
        return x_r
    

class VP_SDE(Transport):
    def __init__(self, beta_min=0.1, beta_d=19.9, epsilon_t=1e-5, T=1000, sigma_d=1.0, enhance_target=False, w_gt=1.0, w_cond=0.0, w_start=0.0, w_end=1.0):
        '''
        Variance preserving (VP) formulation from the paper:
        "Score-Based Generative Modeling through Stochastic Differential Equations".
        '''
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.epsilon_t = epsilon_t
        self.T = T
        super().__init__(sigma_d, 1.0, epsilon_t, enhance_target, w_gt, w_cond, w_start, w_end)
        
    def interpolant(self, t: torch.Tensor):
        '''
        The d_alpha_t and d_sigma_t are easy to obtain: 
        # from sympy import *
        # from scipy.stats import *
        # t, beta_d, beta_min = symbols('t beta_d beta_min')
        # sigma = sqrt(exp(0.5 * beta_d * (t ** 2) + beta_min * t) - 1)
        # d_sigma_d_t = diff(sigma, t)
        # print(d_sigma_d_t)
        # sigma = symbols('sigma')
        # alpha_t = (sigma**2 + 1) ** (-0.5)
        # sigma_t = sigma * (sigma**2 + 1) ** (-0.5)
        # d_alpha_d_sigma = diff(alpha_t, sigma)
        # print(d_alpha_d_sigma)
        # d_sigma_d_sigma = diff(sigma_t, sigma)
        # print(d_sigma_d_sigma)
        '''
        beta_t = self.beta(t)
        alpha_t = 1 / torch.sqrt(beta_t**2 + 1) 
        sigma_t = beta_t / torch.sqrt(beta_t**2 + 1)
        d_alpha_t = -0.5 * (self.beta_d * t + self.beta_min) / (beta_t**2 + 1).sqrt()
        d_sigma_t =  0.5 * (self.beta_d * t + self.beta_min) / (beta_t * (beta_t**2 + 1).sqrt())
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def beta(self, t: torch.Tensor):
        return torch.sqrt((0.5 * self.beta_d * (t ** 2) + self. beta_min * t).exp() - 1)

    def sample_t(self, batch_size, dtype, device):
        rnd_uniform = torch.rand((batch_size, ), dtype=dtype, device=device)
        t = 1 + rnd_uniform * (self.epsilon_t - 1) # [epsilon_t, 1]
        return t
    
    def c_noise(self, t: torch.Tensor):
        return (self.T - 1) * t 
    
    def target(
        self, 
        x_t: torch.Tensor, 
        v_t: torch.Tensor, 
        x: torch.Tensor, 
        z: torch.Tensor, 
        t: torch.Tensor, 
        r: torch.Tensor, 
        dF_dv_dt: torch.Tensor,
        F_t_cond: Optional[torch.Tensor] = 0.0, 
        F_t_uncond: Optional[torch.Tensor] = 0.0, 
        enhance_target = False,
    ):
        if enhance_target:
            w_gt = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_gt, 1.0)
            w_cond = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_cond, 0.0)
            z = w_gt * z + w_cond * F_t_cond + (1-w_gt-w_cond) * F_t_uncond
        beta_t = self.beta(t)
        beta_r = self.beta(r)
        d_beta_t = (self.beta_d * t + self.beta_min) * (beta_t ** 2 + 1) / (2 * beta_t)
        F_target = z - dF_dv_dt * (beta_t - beta_r) / d_beta_t
        return F_target
    
    def from_x_t_to_x_r(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor, F: torch.Tensor, s_ratio=0.0):
        beta_t = self.beta(t)
        beta_r = self.beta(r)
        A_t = (beta_t ** 2 + 1).sqrt() / (beta_r ** 2 + 1).sqrt()
        B_t = (beta_r - beta_t) / (beta_r ** 2 + 1).sqrt()
        x_r = A_t * x_t + B_t * F
        return x_r




class VE_SDE(Transport):
    def __init__(self, sigma_min=0.02, sigma_max=100, sigma_d=1.0, enhance_target=False, w_gt=1.0, w_cond=0.0, w_start=0.0, w_end=1.0):
        '''
        Variance exploding (VE) formulation from the paper: 
        "Score-Based Generative Modeling through Stochastic Differential Equations".
        '''
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        super().__init__(sigma_d, sigma_max, sigma_min, enhance_target, w_gt, w_cond, w_start, w_end)
        
    def interpolant(self, t: torch.Tensor):
        alpha_t = 1 
        sigma_t = t
        d_alpha_t =  0
        d_sigma_t = 1
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_t(self, batch_size, dtype, device):
        rnd_uniform = torch.rand((batch_size, ), dtype=dtype, device=device)
        t = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform) # [sigma_min, sigma_max]
        return t
    
    def c_noise(self, t: torch.Tensor):
        return torch.log(0.5 * t)
    
    def target(
        self, 
        x_t: torch.Tensor, 
        v_t: torch.Tensor, 
        x: torch.Tensor, 
        z: torch.Tensor, 
        t: torch.Tensor, 
        r: torch.Tensor, 
        dF_dv_dt: torch.Tensor,
        F_t_cond: Optional[torch.Tensor] = 0.0, 
        F_t_uncond: Optional[torch.Tensor] = 0.0, 
        enhance_target = False,
    ):
        if enhance_target:
            w_gt = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_gt, 1.0)
            w_cond = torch.where((t>=self.w_start) & (t<=self.w_end), self.w_cond, 0.0)
            z = w_gt * z + w_cond * (-F_t_cond) + (1-w_gt-w_cond) * (-F_t_uncond)
        F_target = (r - t) * dF_dv_dt - z
        return F_target
    

    def from_x_t_to_x_r(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor, F: torch.Tensor, s_ratio=0.0):
        x_r = x_t + (t - r) * F
        return x_r
