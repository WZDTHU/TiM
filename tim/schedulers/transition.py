import torch
import torch.nn.functional as F
from copy import deepcopy
from .transports import Transport
from tim.models.utils.funcs import expand_t_like_x


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

    

class TransitionSchedule:
    def __init__(
            self,
            transport: Transport, 
            diffusion_ratio: float = 0.0,
            consistency_ratio: float = 0.0,
            derivative_type: str = 'dde',
            differential_epsilon: float = 0.005,
            weight_t_and_r: bool = True,
            weight_time_type: str = 'constant',
            weight_time_tangent: bool=False,
            weight_time_sigmoid: bool=False,
        ):
        self.transport = transport
        self.diffusion_ratio = diffusion_ratio
        self.consistency_ratio = consistency_ratio
        self.derivative_type = derivative_type
        self.differential_epsilon = differential_epsilon
        self.weight_t_and_r = weight_t_and_r
        self.weight_time_type = weight_time_type
        self.weight_time_tangent = weight_time_tangent
        self.weight_time_sigmoid = weight_time_sigmoid
        
    
    def sample_t_and_r(self, batch_size, dtype, device):
        t_1 = self.transport.sample_t(batch_size=batch_size, dtype=dtype, device=device)
        t_2 = self.transport.sample_t(batch_size=batch_size, dtype=dtype, device=device)
        # t is the larger one, and r is the smaller one
        t = torch.maximum(t_1, t_2)
        r = torch.minimum(t_1, t_2)
        # some samples with t=r, corresponding to diffusion training
        n_diffusion = round(self.diffusion_ratio * len(t))
        r[: n_diffusion] = t[: n_diffusion]
        # some samples with r=0, corresponding to consistency training
        n_consistency = round(self.consistency_ratio * len(t))
        if n_consistency != 0:
            r[-n_consistency: ] = self.transport.T_min
        return t, r, n_diffusion
    
    def prepare_input(self, batch_size, x, z):
        # sample timestep according to log-normal distribution of sigmas following EDM
        t, r, n_diffusion = self.sample_t_and_r(batch_size=batch_size, dtype=x.dtype, device=x.device)
        # reshape (B, ) -> (B, 1, 1, 1)
        t, r = expand_t_like_x(t, x), expand_t_like_x(r, x)
        # prepere inputs
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.transport.interpolant(t)
        x_t = alpha_t * x + sigma_t * z
        v_t = d_alpha_t * x + d_sigma_t * z
        return x_t, v_t, t, r, n_diffusion

    
    def model_forward(self, model, x_t, t, r, model_kwargs, rng_state):
        # model_input
        t_input = self.transport.c_noise(t.flatten())
        r_input = self.transport.c_noise(r.flatten())
        # model_output
        torch.cuda.set_rng_state(rng_state)
        model_output = model(x_t, t_input, r_input, **model_kwargs)
        return model_output

    @torch.no_grad()
    def jvp_derivative(self, model, x_t, v_t, t, r, model_kwargs, rng_state, n_diffusion):
        if n_diffusion == x_t.size(0):
            return 0
        _dF_dv_dt = torch.zeros_like(x_t)
        # only calculate the dF_dv_dt when t!=r
        x_t, v_t, t, r = x_t[n_diffusion: ], v_t[n_diffusion: ], t[n_diffusion: ], r[n_diffusion: ]
        for k, v in model_kwargs.items():
            if type(v) == torch.Tensor:
                model_kwargs[k] = model_kwargs[k][n_diffusion: ]
        model_kwargs['return_zs'] = False
        def model_jvp(x_t, t, r):
            model_kwargs['attn_type'] = 'vanilla_attn'
            model_kwargs['jvp'] = True
            t_input = self.transport.c_noise(t.flatten())
            r_input = self.transport.c_noise(r.flatten())
            return model(x_t, t_input, r_input, **model_kwargs)
        torch.cuda.set_rng_state(rng_state)
        F_pred, dF_dv_dt = torch.func.jvp(
            lambda x_t, t, r: model_jvp(x_t, t, r),
            (x_t, t, r),
            (v_t, torch.ones_like(t), torch.zeros_like(r))
        )
        _dF_dv_dt[n_diffusion: ] = dF_dv_dt
        return _dF_dv_dt
    
    @torch.no_grad()
    def diff_derivative(self, model, x, z, t, r, model_kwargs, rng_state, n_diffusion):
        if n_diffusion == x.size(0):
            return 0
        _dF_dv_dt = torch.zeros_like(x)
        # only calculate the dF_dv_dt when t!=r
        x, z, t, r = x[n_diffusion: ], z[n_diffusion: ], t[n_diffusion: ], r[n_diffusion: ]
        for k, v in model_kwargs.items():
            if type(v) == torch.Tensor:
                model_kwargs[k] = model_kwargs[k][n_diffusion: ]
        model_kwargs['return_zs'] = False
        model_kwargs['jvp'] = True
    
        def xfunc(t):
            alpha_t, sigma_t, _, _ = self.transport.interpolant(t)
            x_t = alpha_t * x + sigma_t * z
            return self.model_forward(model, x_t, t, r, model_kwargs, rng_state)
        epsilon = self.differential_epsilon
        fc1_dt = 1 / (2 * epsilon)
        dF_dv_dt = xfunc(t + epsilon) * fc1_dt - xfunc(t - epsilon) * fc1_dt
        _dF_dv_dt[n_diffusion: ] = dF_dv_dt
        return _dF_dv_dt

    def get_enhanced_target(self, model, x_t, t, model_kwargs, null_kwargs, rng_state):
        with torch.no_grad():
            t_input = self.transport.c_noise(t.flatten())
            if self.transport.w_cond > 0:
                F_t_cond = self.model_forward(model, x_t, t_input, t_input, model_kwargs, rng_state)
            else: 
                F_t_cond = 0
            F_t_uncond = self.model_forward(model, x_t, t_input, t_input, null_kwargs, rng_state)
        return F_t_cond, F_t_uncond

    def time_weighting(self, t, r, n_diffusion):
        if self.weight_time_tangent:
            t, r = torch.tan(t), torch.tan(r)
        elif self.weight_time_sigmoid:
            t, r = t/(1-t), r/(1-r)
        if self.weight_t_and_r:
            delta_t = (t - r).flatten()
        else: 
            delta_t = t.flatten()
        if self.weight_time_type == 'constant':
            weight = torch.ones_like(delta_t)
        elif self.weight_time_type == 'reciprocal':
            weight = 1 / (delta_t + self.transport.sigma_d)
        elif self.weight_time_type == 'sqrt':
            weight = 1 / (delta_t + self.transport.sigma_d).sqrt()
        elif self.weight_time_type == 'square':
            weight = 1 / (delta_t + self.transport.sigma_d)**2
        elif self.weight_time_type == 'Soft-Min-SNR':
            weight = 1 / (delta_t ** 2 + self.transport.sigma_d ** 2)
        else:
            raise NotImplementedError
        weight[: n_diffusion] = 1.0
        return weight
        

    def adaptive_weighting(self, loss, eps=10e-6):
        weight = 1 / (loss.detach() + eps)
        return weight


    def __call__(
            self, 
            model, 
            ema_model, 
            unwrapped_model, 
            batch_size, 
            x, 
            z, 
            model_kwargs, 
            use_dir_loss=False, 
            h_target=None,
            ema_kwargs={},
            null_kwargs={},
        ):
        # prepare model input
        x_t, v_t, t, r, n_diffusion = self.prepare_input(batch_size, x, z)

        rng_state = torch.cuda.get_rng_state()
        # get prediction
        F_pred, h_proj = self.model_forward(model, x_t, t, r, model_kwargs, rng_state)
        # get target
        if self.derivative_type == 'jvp':
            dF_dv_dt = self.jvp_derivative(unwrapped_model, x_t, v_t, t, r, model_kwargs, rng_state, n_diffusion)
        else:
            dF_dv_dt = self.diff_derivative(unwrapped_model, x, z, t, r, model_kwargs, rng_state, n_diffusion)
        
        if self.transport.enhance_target:
            F_t_cond, F_t_uncond = self.get_enhanced_target(ema_model, x_t, t, ema_kwargs, null_kwargs, rng_state)
            enhance_target = True
        else:
            F_t_cond, F_t_uncond, enhance_target = 0, 0, False
        F_target = self.transport.target(x_t, v_t, x, z, t, r, dF_dv_dt, F_t_cond, F_t_uncond, enhance_target)
        denoising_loss = mean_flat((F_pred - F_target) ** 2)
        denoising_loss = torch.nan_to_num(denoising_loss, nan=0, posinf=1e5, neginf=-1e5)

        if use_dir_loss:
            directional_loss = mean_flat(1 - F.cosine_similarity(F_pred, F_target, dim=1))
            directional_loss = torch.nan_to_num(directional_loss, nan=0, posinf=1e5, neginf=-1e5)
            denoising_loss += directional_loss

        weight = self.time_weighting(t, r, n_diffusion) * self.adaptive_weighting(denoising_loss)
        weighted_loss = weight * denoising_loss
        weighted_loss = weighted_loss.mean()
        
        proj_loss = mean_flat(1 - torch.cosine_similarity(h_proj, h_target, dim=-1))
        proj_loss = torch.nan_to_num(proj_loss, nan=0, posinf=1e5, neginf=-1e5)
        proj_loss = proj_loss.mean()

        loss_dict = dict(
            weighted_loss = weighted_loss.detach().item(),
            denoising_loss = denoising_loss.mean().detach().item(),
            proj_loss = proj_loss.detach().item(),
        )

        return weighted_loss, proj_loss, loss_dict

    
    def forward_with_cfg(self, model, x_t, t, r, y, y_null, cfg_scale, cfg_low, cfg_high):
        apply_cfg = cfg_scale > 1.0 and t > cfg_low and t < cfg_high
        if apply_cfg:
            x_cur = torch.cat([x_t] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            x_cur = x_t
            y_cur = y
        t_cur = torch.ones(x_cur.size(0)).to(x_cur) * self.transport.c_noise(t)
        r_cur = torch.ones(x_cur.size(0)).to(x_cur) * self.transport.c_noise(r)
        F_pred = model(x_cur, t_cur, r_cur, y_cur)
        if apply_cfg:
            F_cond, F_uncond = F_pred.chunk(2)
            F_pred = F_uncond + cfg_scale * (F_cond - F_uncond)   
        return F_pred


    @torch.no_grad()
    def sample(self, 
        model, 
        y,
        y_null,
        z, 
        T_max,
        T_min=0.0,
        num_steps=4, 
        cfg_scale=1.0, 
        cfg_low=0.0, 
        cfg_high=1.0,
        stochasticity_ratio=0.0,
        sample_type: str = 'transition', # 'transition', diffusion
    ):
        _dtype = z.dtype
        t_steps = torch.linspace(T_max, T_min, num_steps+1, dtype=torch.float64).to(z)
        cfg_low = cfg_low * T_max
        cfg_high = cfg_high * T_max

        x_cur = deepcopy(z).to(torch.float64)
        samples = [z]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # x_{N} -> x_{N-1} -> ... -> x_{n+1} -> x_{n} -> x_{n-1} -> ... -> x_{1} -> x_{0}
            if sample_type == 'transition':
                _t_next = t_next
            elif sample_type == 'ddiffusion':
                _t_next = t_cur
            else:
                raise
            F_pred = self.forward_with_cfg(
                model, x_cur.to(_dtype), t_cur, _t_next, y, y_null, cfg_scale, cfg_low, cfg_high
            ).to(torch.float64)
            if stochasticity_ratio > 0.0 and t_cur < T_max and _t_next > T_min:
                s_ratio = stochasticity_ratio
            else: 
                s_ratio = 0.0
            x_next = self.transport.from_x_t_to_x_r(x_cur, t_cur, t_next, F_pred, s_ratio)     
            samples.append(x_next)
            x_cur = x_next

        return torch.stack(samples, dim=0).to(torch.float32)
            
