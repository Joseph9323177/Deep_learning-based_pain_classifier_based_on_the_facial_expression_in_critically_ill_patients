import torch
from torch import nn
import torch.nn.functional as F
from encoder import Encoder
from facenet_pytorch import InceptionResnetV1

class CLS(nn.Module):
    def __init__(self, arch, pretrained, combine, two_layer, c, reference='none', emb_dim=512, normal=False):
        super(CLS, self).__init__()
        self.encoder = Encoder(arch=arch, pretrained=True) if arch!='inception' else InceptionResnetV1(pretrained=('vggface2' if pretrained else None))
#        self.encoder.load_state_dict(torch.load(f'{arch}_encoder0.ckpt'))
        self.c = c
        self.ref = reference
        if 'all' in self.ref:
            n = self.c
        elif self.ref=='none':
            n = 0
        else:
            n = 1
        dim_combined = emb_dim*(n+1)
        if two_layer:
            self.proj = nn.Sequential(nn.Linear(dim_combined, 256), nn.ReLU(), nn.Dropout())
            self.cls = nn.Linear(256, self.c)
        else:
            self.proj = None
            self.cls = nn.Linear(dim_combined, self.c)
        self.combine = combine
        self.normal = normal
    def forward(self, r, x, no_fc=False):
        x = self.encoder(x)
        if self.ref=='none':
            embs = x
        else:
            r = self.encoder(r)
            if self.ref=='all1':
                r = r.reshape(self.c, -1, r.size()[-1]).mean(dim=1)
            elif self.ref=='all0':
                r = r.reshape(-1, self.c, r.size()[-1]).mean(dim=0)
            else:
                r = r.mean(dim=0)
            r = r.reshape(1, -1).repeat((len(x), 1))
            embs = torch.cat((r, x), dim=-1)
        if self.proj != None:
            embs = self.proj(embs)
        preds = self.cls(embs)
        return preds.reshape(-1, self.c), embs

class FS_CLS(nn.Module):
    def __init__(self, arch, pretrained, combine, two_layer, args):
        super(FS_CLS, self).__init__()
        self.encoder = Encoder(arch=arch, pretrained=True, emb_dim=args.emb_dim) if arch!='inception' else InceptionResnetV1(pretrained=('vggface2' if pretrained else None))
#        self.encoder.load_state_dict(torch.load(f'{arch}_encoder0.ckpt'))
        self.c = args.c
        self.relation = nn.Linear(2*args.emb_dim, 1)
        self.mode = args.few_shot
    def forward(self, r, x):
        if r==None:
            return self.encoder(x), 0
        x = self.encoder(x)
        r = self.encoder(r)
        r = r.reshape(self.c, -1, r.size()[-1]).mean(dim=1)
        if self.mode=='proto':
            predictions = -torch.cdist(x.unsqueeze(0), r.unsqueeze(0))[0]
        else:
            cat = torch.cat((r.repeat(len(x), 1), x.repeat_interleave(self.c, dim=0)), dim=-1)
            predictions = torch.sigmoid(self.relation(cat).reshape(-1, 2))
        return predictions, torch.cat([r, x], dim=0)

class Scorer(nn.Module):
    def __init__(self, arch, pretrained, combine, two_layer, args):
        super(Scorer, self).__init__()
        self.encoder = Encoder(arch=arch, pretrained=True, emb_dim=args.emb_dim) if arch!='inception' else InceptionResnetV1(pretrained=('vggface2' if pretrained else None))
#        self.encoder.load_state_dict(torch.load(f'{arch}_encoder0.ckpt'))
        self.c = args.c
        self.score = nn.Linear(args.emb_dim, 1)
        self.mode = args.few_shot
    def forward(self, x):
        x = self.encoder(x)
        return F.sigmoid(self.score(x))
