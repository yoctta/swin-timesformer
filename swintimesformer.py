import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention
def attn(q, k, v, B):
    sim = einsum('b i d, b j d -> b i j', q, k)+B.unsqueeze(0)
    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        M,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.M=M[0]
        if len(M)==1:
            self.Mat=nn.Parameter(torch.zeros([2*self.M-1]),requires_grad=True)
            B=torch.arange(self.M).view(-1,1)-torch.arange(self.M).view(1,-1)+(self.M-1)
            self.register_buffer('B',B.view(-1))
            self.dim=1
        else:
            self.Mat=nn.Parameter(torch.zeros([(2*self.M-1)**2]),requires_grad=True)
            B=torch.arange(self.M)
            B1,B2=torch.meshgrid(B,B)
            B=torch.stack([B1.reshape(-1),B2.reshape(-1)],dim=1)
            B=B.view(1,self.M**2,2)-B.view(self.M**2,1,2)+torch.tensor([[[self.M-1,self.M-1]]])
            B=B[:,:,0]*(2*self.M-1)+B[:,:,1]
            self.register_buffer('B',B.view(-1))
            self.dim=2

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale
        if self.dim==1:
            B=self.Mat.index_select(0,self.B).view(self.M,self.M)
        else:
            B=self.Mat.index_select(0,self.B).view(self.M**2,self.M**2)
        out = attn(q, k, v,B)
        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        # combine heads out
        return self.to_out(out)

# main classes
class DownSample(nn.Module):
    def __init__(self,dim,stride):
        super().__init__()
        self.fc1=nn.Linear(dim*stride**3,dim*2)
        self.stride=stride
    def forward(self,x):
        b,t,h,w,d=x.shape
        x=x[:,:t//self.stride*self.stride,:h//self.stride*self.stride,:w//self.stride*self.stride,:]
        return self.fc1(rearrange(x,'b (f M) (h M1) (w M2) d -> b f h w (M M1 M2 d)',M=self.stride,M1=self.stride,M2=self.stride))

class SwinBlock(nn.Module):
    def __init__(self,dim,space_M,time_M,heads,dim_head,attn_dropout,ff_dropout):
        super().__init__()
        self.time_M=time_M
        self.space_M=space_M
        self.time_attn1=PreNorm(dim, Attention(dim,[time_M], dim_head = dim_head, heads = heads, dropout = attn_dropout))
        self.spatial_attn1=PreNorm(dim, Attention(dim,[space_M,0], dim_head = dim_head, heads = heads, dropout = attn_dropout))
        self.ff1=PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
        self.time_attn2=PreNorm(dim, Attention(dim,[time_M], dim_head = dim_head, heads = heads, dropout = attn_dropout))
        self.spatial_attn2=PreNorm(dim, Attention(dim,[space_M,0], dim_head = dim_head, heads = heads, dropout = attn_dropout))
        self.ff2=PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
    
    def forward(self,x):
        x=pad(x,self.time_M,self.space_M)
        x_=x
        x=rearrange(x,'b (n M) h w d -> b n h w M d',M=self.time_M)
        b,n,h,w,m,d=x.shape
        x=rearrange(x,'b n h w m d -> (b n h w) m d')
        x =rearrange(self.time_attn1(x),'(b n h w) m d -> b (n m) h w d',b=b,n=n,h=h,w=w) + x_
        x_=x
        x=rearrange(x,'b t (h M1) (w M2) d -> b t h w (M1 M2) d',M1=self.space_M,M2=self.space_M)
        b,t,h,w,m,d=x.shape
        x=rearrange(x,'b t h w m d -> (b t h w) m d')
        x = rearrange(self.spatial_attn1(x),'(b t h w) (M1 M2) d -> b t (h M1) (w M2) d',b=b,t=t,h=h,w=w,M1=self.space_M,M2=self.space_M) + x_
        x = self.ff1(x) + x
        x=shift(x,[self.time_M//2,self.space_M//2,self.space_M//2])
        x_=x
        x=rearrange(x,'b (n M) h w d -> b n h w M d',M=self.time_M)
        b,n,h,w,m,d=x.shape
        x=rearrange(x,'b n h w m d -> (b n h w) m d')
        x = rearrange(self.time_attn2(x),'(b n h w) m d -> b (n m) h w d',b=b,n=n,h=h,w=w) + x_
        x_=x
        x=rearrange(x,'b t (h M1) (w M2) d -> b t h w (M1 M2) d',M1=self.space_M,M2=self.space_M)
        b,t,h,w,m,d=x.shape
        x=rearrange(x,'b t h w m d -> (b t h w) m d')
        x = rearrange(self.spatial_attn2(x),'(b t h w) (M1 M2) d -> b t (h M1) (w M2) d',b=b,t=t,h=h,w=w,M1=self.space_M,M2=self.space_M) + x_
        x = self.ff2(x) + x_
        x=shift(x,[-self.time_M//2,-self.space_M//2,-self.space_M//2])
        return x


def shift(x,offset):
    B,T,H,W,D=x.shape
    x=rearrange(x,'b t h w d -> b (t h w) d')
    Z=repeat(torch.arange(T,device=x.device),'a -> a H W',H=H,W=W)
    X=repeat(torch.arange(H,device=x.device),'a -> T a W',T=T,W=W)
    Y=repeat(torch.arange(W,device=x.device),'a -> T H a',T=T,H=H)
    Z=(Z+offset[0])%T
    X=(X+offset[1])%H
    Y=(Y+offset[2])%W
    TXY=(Z*H*W+X*W+Y).view(-1)
    x=x.index_select(1,TXY)
    x=rearrange(x,'b (t h w) d -> b t h w d',t=T,h=H,w=W)
    return x

def pad(x,time_M,space_M):
    B,T,H,W,D=x.shape
    t_=(-T)%time_M
    h_=(-H)%space_M
    w_=(-W)%space_M
    x=F.pad(x,[0,t_,0,h_,0,w_,0,0][::-1])
    return x


class SwinTimeSformer(nn.Module):
    def __init__(
        self,
        dim=128,
        num_classes=2,
        space_M = 7,
        time_M = 7,
        patch_size = 4,
        channels = 3,
        stages = [1,1,9,1],
        heads = 4,
        dim_head = 32,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.embd = nn.Linear(patch_dim, dim)

        self.stages = nn.ModuleList([])
        for i,s in enumerate(stages):
            current_stage=nn.ModuleList([])
            if i>0:
                current_stage.append(DownSample(dim,2))
                dim*=2
            for j in range(stages[i]):
                current_stage.append(SwinBlock(dim,space_M,time_M,heads,dim_head,attn_dropout,ff_dropout))
            self.stages.append(current_stage)
        self.pooling=nn.AdaptiveMaxPool3d((1,1,1))
        self.classifier=nn.Linear(dim,num_classes)

    def forward(self, video):
        pac=rearrange(video,'b c f (h m1) (w m2) -> b f h w (c m1 m2 )',m1=self.patch_size,m2=self.patch_size)
        x=self.embd(pac)
        for i,s in enumerate(self.stages):
            for j in s:
                x=j(x)
        x=self.pooling(rearrange(x,'b f h w d -> b d f h w')).squeeze()
        return self.classifier(x)


    

        


