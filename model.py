import torch
import torch.nn as nn

def extract(a, t, x_shape):
    """
    
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(a, dim=0, index=t)
    assert list(out.shape) == [bs]
    return out.view([bs] + ((len(x_shape) - 1) * [1]))

class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        T: int,
        beta_1: float,
        beta_T: float,
        model_var_type: str = 'fixedlarge'
    ):  
        """
        U-Netを学習させるためのクラス

        Params
        ---
        model : U-net
        T : 推論器の段数
        beta_1, beta_T : 推論器において分散を変化させるパラメータ
            - q(x_t|x_{t-1})=N(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)
            - beta1からbetaTまで等間隔で値を設定する
        model_var_type :
        """
        super().__init__()
        self.model = model
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.model_var_type = model_var_type
        # ここから, 分散に関する定数の用意
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)
        self.register_buffer('betas', betas)
        
        alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([Tensor([1.]), alphas_cumprod[:-1]])
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        posterior_log_variance_clipped = torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]]))
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)

        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)

        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)

        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    def forward(self, x_0, eps_true, t):
        """
        
        Algorithm 1.

        x_0: shape (B, C, H, W), 入力画像
        eps_true : shape (B, C, H, W)
        t: shape (B, ), 時間ステップ（ミニバッチ数の次元をもつ）
        """

        # ジンギス本 式(1.9)
        x_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 \
        + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * eps_true

        # x_tをU-netに入力して, e_\theta(x_t,t) を予測
        eps_pred = self.model(x_t, t)
        # 平均２乗誤差を損失として計算
        loss = F.mse_loss(eps_pred, eps_true, reduction='mean')
        return {
            'loss': loss,
            'eps': eps_pred
        }
    
    def p_sample(self, x_t, i_time):
        """
        x_t : shape (B, C, H, W), 入力画像
        i_time : int, 時間ステップ
        """
        # ミニバッチの数だけコピーしてベクトルにする
        t = x_t.new_ones((x_t.shape[0], ), dtype=torch.long) * i_time
        # \mu_\theta(x_t,t)を計算
        mean, _, log_var, pred_xstart = self._p_mean_variance(x_t=x_t, t=t)
        # no noise when t == 0
        if i_time > 0:
            z = torch.randn_like(x_t)
        else:
            z = 0
        # ジンギス本 式(1.33)
        return (mean + torch.exp(0.5 * log_var) * z, pred_xstart)

    def p_sample_loop(self, x_T):
        """
        Algorithm 2.

        x_T shape (B, C, H, W)
        """
        x_t = x_T
        for i_time in reversed(range(self.T)):
            x_t, _ = self.p_sample(x_t, i_time)
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

    def _p_mean_variance(self, x_t, t, clip_denoised=True):
        """
        x_t: 入力画像
        t: 時間ステップ
        mu_theta(x_t, t), sigma_t を求める関数.
        """
        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
            'fixedlarge': (self.betas, torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]]))),
            'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
        }[self.model_var_type]
        # U-Netに入力
        model_variance = extract(model_variance, t, x_t.shape)
        model_log_variance = extract(model_log_variance, t, x_t.shape)

        model_output = self.model(x_t, t)
        _maybe_clip = lambda x_: torch.clip(x_, -1., 1.) if clip_denoised else x_
        # 
        pred_xstart = _maybe_clip(self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output))
        model_mean, _, _ = self._q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)

        return model_mean, model_variance, model_log_variance, pred_xstart

    def _q_posterior_mean_variance(self, x_start, x_t, t):
        """
        式(1.18)
        q(x_{t-1} | x_t, x_0)の平均と分散を求める関数.
        """
        assert x_start.size() == x_t.size()
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        式(1.29)の第二引数の形にする補助関数
        - x_t と eps から x_0 を求める関数.
        """
        assert x_t.size() == eps.size(), (x_t.size(), eps.size())
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t \
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps