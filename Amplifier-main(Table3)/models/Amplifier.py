import torch
import torch.nn as nn
from layers.RevIN import RevIN
from scipy.special import gamma
from torch.nn.parameter import Parameter
from torch.nn import init 
import torch
from torch import Tensor
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Fractional_Order_Matrix_Differential_Solver(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input1,w,b,alpha):
        # print(epoch)
        alpha = torch.tensor(alpha)
        ctx.save_for_backward(input1,w,b,alpha)
        outputs = input1@w + b
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        input1,w,b,alpha = ctx.saved_tensors

        x_fractional, w_fractional = Fractional_Order_Matrix_Differential_Solver.Fractional_Order_Matrix_Differential_Linear(input1,w,b,alpha)   

        x_grad = grad_outputs@x_fractional
        w_grad = w_fractional@grad_outputs
        b_grad = grad_outputs.sum(dim=0)

        return x_grad, w_grad, b_grad,None

    @staticmethod
    def Fractional_Order_Matrix_Differential_Linear(xs,ws,b,alpha):
        wf = ws[:,0].view(1,-1)
        #main
        w_main = torch.mul(xs,(torch.abs(wf)+1e-8)**(1-alpha)/gamma(2-alpha))
        #partial
        w_partial = torch.mul((xs@wf.T).expand(xs.shape) - torch.mul(xs,wf) + b[0], torch.sgn(wf)*(torch.abs(wf)+1e-8)**(-alpha)/gamma(1-alpha))
        return ws.T, (w_main + w_partial).transpose(-2,-1)

class FLinear(nn.Module):
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, alpha=0.9, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    # def forward(self, x, epoch):
    def forward(self, x):
        # print("epoch".format(alpha))
        return Fractional_Order_Matrix_Differential_Solver.apply(x, self.weight, self.bias, self.alpha)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.hidden_size = configs.hidden_size
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.alpha = configs.alpha

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        self.mask_matrix = nn.Parameter(torch.ones(int(self.seq_len / 2) + 1, self.channels))
        self.freq_linear = nn.Linear(int(self.seq_len / 2) + 1, int(self.pred_len / 2) + 1).to(torch.cfloat)

        self.linear_seasonal = nn.Sequential(
            FLinear(self.seq_len, self.hidden_size,self.alpha),
            nn.LeakyReLU(),
            FLinear(self.hidden_size, self.pred_len,self.alpha)
        )

        self.linear_trend = nn.Sequential(
            FLinear(self.seq_len, self.hidden_size,self.alpha),
            nn.LeakyReLU(),
            FLinear(self.hidden_size, self.pred_len,self.alpha)
        )

        # SCI block
        self.SCI = configs.SCI
        self.extract_common_pattern = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.LeakyReLU(),
            nn.Linear(self.channels, 1)
        )

        self.model_common_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len)
        )

        self.model_spacific_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len)
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, C = x.size()

        # RevIN
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        # Energy Amplification Block
        x_fft = torch.fft.rfft(x, dim=1)   # domain conversion
        x_inverse_fft = torch.flip(x_fft, dims=[1])  # flip the spectrum
        x_inverse_fft = x_inverse_fft * self.mask_matrix
        x_amplifier_fft = x_fft + x_inverse_fft
        x_amplifier = torch.fft.irfft(x_amplifier_fft, dim=1)

        # SCI block
        if self.SCI:
            x = x_amplifier
            # extract common pattern
            common_pattern = self.extract_common_pattern(x)
            common_pattern = self.model_common_pattern(common_pattern.permute(0, 2, 1)).permute(0, 2, 1)
            # model specific pattern
            specififc_pattern = x - common_pattern.repeat(1, 1, C)
            specififc_pattern = self.model_spacific_pattern(specififc_pattern.permute(0, 2, 1)).permute(0, 2, 1)

            x = specififc_pattern + common_pattern.repeat(1, 1, C)
            x_amplifier = x

        # Seasonal Trend Forecaster
        seasonal, trend = self.decompsition(x_amplifier)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out_amplifier = seasonal + trend

        # Energy Restoration Block
        out_amplifier_fft = torch.fft.rfft(out_amplifier, dim=1)
        x_inverse_fft = self.freq_linear(x_inverse_fft.permute(0, 2, 1)).permute(0, 2, 1)
        out_fft = out_amplifier_fft - x_inverse_fft
        out = torch.fft.irfft(out_fft, dim=1)

        # inverse RevIN
        z = out
        z = self.revin_layer(z, 'denorm')
        out = z

        return out
