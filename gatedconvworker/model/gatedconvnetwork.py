from gatedconvworker.model.gatedconvlayer import *
from torch import nn
from functools import partial

class Generator(nn.Module):
    def __init__(self, input_cnum = 5):
        super().__init__()

        cnum = 48
        self.tanh = nn.Tanh()
        # stage 1
        self.s1_conv1 = Gen_Conv(input_cnum, cnum, 5, 1)
        self.s1_conv2 = Gen_Conv(cnum // 2, cnum * 2, 3, 2) #downsampling
        self.s1_conv3 = Gen_Conv(cnum, cnum * 2, 3, 1)
        self.s1_conv4 = Gen_Conv(cnum, cnum * 4, 3, 2)
        self.s1_conv5 = Gen_Conv(cnum * 2, cnum * 4, 3, 1)
        self.s1_conv6 = Gen_Conv(cnum * 2, cnum * 4, 3, 1)
        self.s1_conv7 = Gen_Conv(cnum * 2, cnum * 4, 3, dilation_rate=2) #atrous
        self.s1_conv8 = Gen_Conv(cnum * 2, cnum * 4, 3, dilation_rate=4) #atrous
        self.s1_conv9 = Gen_Conv(cnum * 2, cnum * 4, 3, dilation_rate=8) #atrous
        self.s1_conv10 = Gen_Conv(cnum * 2, cnum * 4, 3, dilation_rate=16) #atrous
        self.s1_conv11 = Gen_Conv(cnum * 2, cnum * 4, 3, 1)
        self.s1_conv12 = Gen_Conv(cnum * 2, cnum * 4, 3, 1)
        self.s1_conv13 = Gen_DeConv(cnum * 2, cnum * 2) #upsample
        self.s1_conv14 = Gen_Conv(cnum, cnum * 2, 3, 1)
        self.s1_conv15 = Gen_DeConv(cnum, cnum) #upsample
        self.s1_conv16 = Gen_Conv(cnum // 2, cnum // 2, 3, 1)
        self.s1_conv17 = Gen_Conv(cnum // 4, 3, 3, 1, activation=None)

        # stage 2
        # conv branch
        self.s2_xconv1 = Gen_Conv(3, cnum, 5, 1)
        self.s2_xconv2 = Gen_Conv(cnum // 2, cnum, 3, 2) #downsample
        self.s2_xconv3 = Gen_Conv(cnum // 2, cnum * 2, 3, 1)
        self.s2_xconv4 = Gen_Conv(cnum, cnum * 2, 3, 2) #downsample
        self.s2_xconv5 = Gen_Conv(cnum, cnum * 4, 3, 1)
        self.s2_xconv6 = Gen_Conv(cnum * 2, cnum * 4, 3, 1)
        self.s2_xconv7 = Gen_Conv(cnum * 2, cnum * 4, 3, dilation_rate=2) #atrous
        self.s2_xconv8 = Gen_Conv(cnum * 2, cnum * 4, 3, dilation_rate=4) #atrous
        self.s2_xconv9 = Gen_Conv(cnum * 2, cnum * 4, 3, dilation_rate=8) #atrous
        self.s2_xconv10 = Gen_Conv(cnum * 2, cnum * 4, 3, dilation_rate=16) #atrous

        # attention branch
        self.s2_pmconv1 = Gen_Conv(3, cnum, 5, 1)
        self.s2_pmconv2 = Gen_Conv(cnum // 2, cnum, 3, 2) #downsample
        self.s2_pmconv3 = Gen_Conv(cnum // 2, cnum * 2, 3, 1)
        self.s2_pmconv4 = Gen_Conv(cnum, cnum * 4, 3, 2) #downsample
        self.s2_pmconv5 = Gen_Conv(cnum * 2, cnum * 4, 3, 1)
        self.s2_pmconv6 = Gen_Conv(cnum * 2, cnum * 4, 3, 1, activation=nn.ReLU())
        self.s2_contextual_attention = ContextualAttention(kernel_size=3, stride=1, dilation_rate=2, fuse_k=3, softmax_scale=10, n_down=2)
        self.s2_pmconv9 = Gen_Conv(cnum * 2, cnum * 4, 3, 1)
        self.s2_pmconv10 = Gen_Conv(cnum * 2, cnum * 4, 3, 1)

        #concatenation happens here
        self.s2_allconv11 = Gen_Conv(cnum * 4, cnum * 4, 3, 1)
        self.s2_allconv12 = Gen_Conv(cnum * 2, cnum * 4, 3, 1)
        self.s2_allconv13 = Gen_DeConv(cnum * 2, cnum * 2) #upsample
        self.s2_allconv14 = Gen_Conv(cnum, cnum * 2, 3, 1)
        self.s2_allconv15 = Gen_DeConv(cnum, cnum) #upsample
        self.s2_allconv16 = Gen_Conv(cnum // 2, cnum // 2, 3, 1)
        self.s2_allconv17 = Gen_Conv(cnum // 4, 3, 3, 1, activation=None)

    def forward(self, x, mask):
        xin = x

        #stage 1
        for i in range(1, 18):
            x = getattr(self, "s1_conv" + str(i))(x)
        x = self.tanh(x)
        x_stage1 = x

        #print(x.shape, mask.shape, xin.shape)
        # stage2, paste result as input
        x = x*mask + xin[:, 0:3, :, :]*(1.-mask) #The image channels are in the dimension 1 instead of 3 in tensorflow, explaining the differences in comparison to original code

        # conv branch
        xnow = x
        for i in range(1, 11):
            x = getattr(self, "s2_xconv" + str(i))(x)
        x_hallu = x

        # attenuation branch
        x = xnow
        for i in range(1, 7):
            x = getattr(self, "s2_pmconv" + str(i))(x)
        #print ("before contextual_attention", x.shape)
        x = self.s2_contextual_attention(x, x, mask) 
        #print("after contextual attention", x.shape)
        for i in range(9, 11):
            x = getattr(self, "s2_pmconv" + str(i))(x)
        x_pm = x

        #concatenation
        x = torch.cat([x_hallu, x_pm], dim=1)

        for i in range(11, 18):
            x = getattr(self, "s2_allconv" + str(i))(x)
        x_stage2 = x

        return x_stage1, x_stage2
class ContextualAttention(nn.Module):
    #source https://github.com/JiahuiYu/generative_inpainting
    def __init__(self, kernel_size=3, stride=1, dilation_rate=1, fuse_k=3, softmax_scale=10., n_down=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.n_down = n_down
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, f, b, mask=None):
        raw_int_fs, raw_int_bs = list(f.size()), list(b.size())

        kernel = 2 * self.dilation_rate

        raw_w = extract_image_patches(b, kernel, self.dilation_rate*self.stride, 1)
        
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        f = downsample(f, self.dilation_rate)
        b = downsample(b, self.dilation_rate)
        int_fs, int_bs = list(f.size()), list(b.size())   
        
        f_groups = torch.split(f, 1, dim=0)
        
        w = extract_image_patches(b, self.kernel_size, self.stride, 1)
        
        w = w.view(int_bs[0], int_bs[1], self.kernel_size, self.kernel_size, -1)
        w = w.permute(0, 4, 1, 2, 3)    
        w_groups = torch.split(w, 1, dim=0)

        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]], device=self.device)
        else:
            mask = downsample(mask, (2**self.n_down)*self.dilation_rate)
        int_ms = list(mask.size())
        
        m = extract_image_patches(mask, self.kernel_size, self.stride, self.dilation_rate)
        
        m = m.view(int_ms[0], int_ms[1], self.kernel_size, self.kernel_size, -1)
        m = m.permute(0, 4, 1, 2, 3)    
        m = m[0]    
        
        mm = (torch.mean(m, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3)  

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    
        fuse_weight = torch.eye(k, device=self.device).view(1, 1, k, k)  # 1*1*k*k

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            wi = wi[0]  
            wi_normed = wi / torch.sqrt(torch.sum(torch.pow(wi, 2), dim=[1, 2, 3], keepdim=True)).clamp_min(1e-4)
            
            xi = padding_function(xi, self.kernel_size, 1, 1) 
            yi = F.conv2d(xi, wi_normed, stride=1)

            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            
            yi = yi * mm 
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  
           
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.dilation_rate, padding=1) / 4.  
            y.append(yi)
            
        y = torch.cat(y, dim=0) 
        y = y.contiguous().view(raw_int_fs)

        return y

def downsample(images, n=2): #TODO
    in_height, in_width = images.shape[2:]
    out_height, out_width = in_height // n, in_width // n
    height_inds = torch.linspace(0, in_height-1, steps=out_height, device=images.device).add_(0.5).floor_().long()
    width_inds = torch.linspace(0, in_width-1, steps=out_width, device=images.device).add_(0.5).floor_().long()
    return images[:, :, height_inds][..., width_inds]

def extract_image_patches(image, kernel, stride, dilation_rate):
    image = padding_function(image, kernel, stride, dilation_rate)
    return torch.nn.Unfold(kernel, dilation_rate, 0, stride)(image)

class Discriminator(nn.Module):
    def __init__(self, input_cnum, output_cnum):
        super().__init__()
        self.conv1 = Dis_Conv(input_cnum, output_cnum, 5, 2)
        self.conv2 = Dis_Conv(output_cnum, 2*output_cnum, 5, 2)
        self.conv3 = Dis_Conv(2*output_cnum, 4*output_cnum, 5, 2)
        self.conv4 = Dis_Conv(4*output_cnum, 4*output_cnum, 5, 2)
        self.conv5 = Dis_Conv(4*output_cnum, 4*output_cnum, 5, 2)
        self.conv6 = Dis_Conv(4*output_cnum, 4*output_cnum, 5, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        for i in range(1, 7):
            x = getattr(self, "conv" + str(i))(x)
        x = self.flatten(x)
        return x