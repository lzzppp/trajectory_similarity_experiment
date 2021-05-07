import torch
import torch.nn as nn
import torch.nn.functional as F


class Bypass (nn.Module):
    _supported_styles = ['residual', 'highway']
    
    @classmethod
    def supports_style(cls, style):
        return style.lower () in cls._supported_styles
    
    def __init__(self, style, residual_scale=True, highway_bias=-2, input_size=None):
        super (Bypass, self).__init__ ()
        assert self.supports_style (style)
        self.style = style.lower ()
        self.residual_scale = residual_scale
        self.highway_bias = highway_bias
        self.highway_gate = nn.Linear (input_size[1], input_size[0])
    
    def forward(self, transformed, raw):
        assert transformed.shape[:-1] == raw.shape[:-1]
        
        tsize = transformed.shape[-1]
        rsize = raw.shape[-1]
        adjusted_raw = raw
        if tsize < rsize:
            assert rsize / tsize <= 50
            if rsize % tsize != 0:
                padded = F.pad (raw, (0, tsize - rsize % tsize))
            else:
                padded = raw
            adjusted_raw = padded.view (*raw.shape[:-1], -1, tsize).sum (-2) * math.sqrt (
                tsize / rsize)
        elif tsize > rsize:
            multiples = math.ceil (tsize / rsize)
            adjusted_raw = raw.repeat (*([1] * (raw.dim () - 1)), multiples).narrow (
                -1, 0, tsize)
        
        if self.style == 'residual':
            res = transformed + adjusted_raw
            if self.residual_scale:
                res *= math.sqrt (0.5)
            return res
        elif self.style == 'highway':
            transform_gate = torch.sigmoid (self.highway_gate (raw) + self.highway_bias)
            carry_gate = 1 - transform_gate
            return transform_gate * transformed + carry_gate * adjusted_raw


class Transform (nn.Module):
    _supported_nonlinearities = [
        'sigmoid', 'tanh', 'relu', 'elu', 'selu', 'glu', 'leaky_relu'
    ]
    
    @classmethod
    def supports_nonlinearity(cls, nonlin):
        return nonlin.lower () in cls._supported_nonlinearities
    
    def __init__(self,
                 style,
                 layers=1,
                 bypass_network=None,
                 non_linearity='leaky_relu',
                 hidden_size=None,
                 output_size=None,
                 input_size=None):
        super (Transform, self).__init__ ()
        hidden_size = hidden_size or input_size
        output_size = output_size or hidden_size
        
        parts = style.split ('-')
        
        if 'layer' in parts:
            layers = int (parts[parts.index ('layer') - 1])
        
        for part in parts:
            if Bypass.supports_style (part):
                bypass_network = part
            if Transform.supports_nonlinearity (part):
                non_linearity = part
        
        self.transforms = nn.ModuleList ()
        self.bypass_networks = nn.ModuleList ()
        
        assert (non_linearity is None or self.supports_nonlinearity (non_linearity))
        self.non_linearity = non_linearity.lower () if non_linearity else None
        
        transform_in_size = input_size
        transform_out_size = hidden_size
        
        for layer in range (layers):
            if layer == layers - 1:
                transform_out_size = output_size
            self.transforms.append (nn.Linear (transform_in_size, transform_out_size))
            self.bypass_networks.append (Bypass ("highway", input_size=[hidden_size, hidden_size]))
            transform_in_size = transform_out_size
    
    def forward(self, input_data):
        output = input_data
        
        for transform, bypass in zip (self.transforms, self.bypass_networks):
            new_output = transform (output)
            if self.non_linearity:
                new_output = getattr (F, self.non_linearity) (new_output)
            if bypass:
                new_output = bypass (new_output, output)
            output = new_output
        
        return output


class f_hid_sim (nn.Module):
    def __init__(self, hidden_size):
        super (f_hid_sim, self).__init__ ()
        self.linear_i = nn.Linear (hidden_size, hidden_size)
        self.linear_c = nn.Linear (hidden_size, hidden_size)
        self.linear_o = nn.Linear (hidden_size, hidden_size)
        self.init_weight ()
    
    def init_weight(self):
        nn.init.uniform_ (self.linear_i.weight)
        nn.init.uniform_ (self.linear_o.weight)
        nn.init.uniform_ (self.linear_c.weight)
        nn.init.zeros_ (self.linear_i.bias)
        nn.init.zeros_ (self.linear_o.bias)
        nn.init.zeros_ (self.linear_c.bias)
    
    def forward(self, hi):
        C = torch.mul (torch.sigmoid (self.linear_i (hi)),
                       torch.tanh (self.linear_c (hi)))
        hi_ = torch.mul (torch.sigmoid (self.linear_o (hi)),
                         torch.tanh (C))
        output = hi_ + hi
        return output
    
class HighwayMLP(nn.Module):
    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)