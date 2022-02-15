import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import random
from encoder import Encoder
from facenet_pytorch import InceptionResnetV1


class LSTM(nn.Module):
    def __init__(self, args, path=None):
        super(LSTM, self).__init__()
        self.z = args.arch=='resnet34'
        dim_combined = (args.emb_dim if args.no_fc else 3)*(2 if self.z else (args.c+1))
        self.encoder = CLS(args.arch, args.pretrained, args.combine, args.two_layer, args)
        if path:
            print('load pretrained CNN')
            self.encoder.load_state_dict(torch.load(path))
        dim_output = 32 if dim_combined>20 else 8
        self.lstm = nn.LSTM(dim_combined, dim_output, bidirectional=args.bi)
        self.fc = nn.Linear(dim_output*(2 if args.bi else 1), args.c)
        self.bi = args.bi
        self.c = args.c
        self.no_fc = args.no_fc
        self.dropout = nn.Dropout(0.5)
    def forward(self, r, x):
        hidden = self.encoder(r, x, self.no_fc)
        hidden = torch.transpose(hidden.reshape(-1, 50, hidden.size()[-1]), 0, 1)
        ouptput, (c, h) = self.lstm(self.dropout(hidden))
        last_h = torch.cat([h[-1], h[-2]], dim=-1) if self.bi else h[-1]
        return self.fc(self.dropout(last_h))

class LSTM3(nn.Module):
    def __init__(self, args, paths=None):
        super(LSTM3, self).__init__()
        dim_combined = 4096 if args.no_fc else 9#)*(2 if args.z else (args.c+1))
        self.encoders = nn.ModuleList([CLS(arch, False, args.combine, args.two_layer, args) for arch in ['resnet34', 'vgg16', 'inception']])
        if paths:
            print('load pretrained CNN')
            for i in range(3):
                print(i)
                self.encoders[i].load_state_dict(torch.load(paths[i]))
        dim_output = args.dim_lstm
        self.lstm = nn.LSTM(dim_combined, dim_output, bidirectional=args.bi)
        self.fc = nn.Linear(dim_output*(2 if args.bi else 1), args.c)
        self.bi = args.bi
        self.c = args.c
        self.no_fc = args.no_fc
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)
        print(self.fc)
    def forward(self, rs, xs):
        if self.no_fc:
            hiddens = [self.encoders[i](rs[i], xs[i], self.no_fc) for i in range(3)]
        else:
            hiddens = [self.softmax(self.encoders[i](rs[i], xs[i])) for i in range(3)]
        hiddens = torch.cat([torch.transpose(hidden.reshape(-1, 50, hidden.size()[-1]), 0, 1) for hidden in hiddens], dim=-1)
        ouptput, (c, h) = self.lstm(self.dropout(hiddens))
        last_h = torch.cat([h[-1], h[-2]], dim=-1) if self.bi else h[-1]
        return self.fc(self.dropout(last_h))

class LSTM5(nn.Module):
    def __init__(self, args, paths=None):
        super(LSTM5, self).__init__()
        if args.c == 3:
            dim_combined = 768  # previously was 2816
        elif args.c == 2 and args.no_one == True:
            dim_combined = 768
        else:
            dim_combined = 768 # previously was 2304
        encoders = []
        # self.reference = args.reference # we don't really need this for training
        if args.c == 3:
            self.inception_list = [False, False, True]
            for arch, two_layer, ref in [('vgg16', True, 'zero'), ('resnet34', True, 'zero'),('inception', True, 'zero')]:  # here to adjust model detail
                encoders.append(RefCLS(arch, False, args.combine, two_layer, ref, c=args.c))
        elif args.c == 2:
            if args.no_one == True:
                self.inception_list = [False, False, True]
                for arch, two_layer, ref in [('vgg16', True, 'zero'), ('resnet34', True, 'zero'),('inception', True, 'zero')]:  # here to adjust model detail
                    encoders.append(RefCLS(arch, False, args.combine, two_layer, ref, c=args.c))
            elif args.no_one == False:
                self.inception_list = [False, False, True]
                for arch, two_layer, ref in [('vgg16', True, 'zero'), ('resnet34', True, 'zero'),('inception', True, 'zero')]:  # here to adjust model detail
                    encoders.append(RefCLS(arch, False, args.combine, two_layer, ref, c=args.c))

        self.encoders = nn.ModuleList(encoders)
        if paths:
            print('load pretrained CNN')
            for i in range(len(paths)):
                self.encoders[i].load_state_dict(torch.load(paths[i])) # the shape of weight must match to pretrained model
        dim_output = args.dim_lstm
        self.lstm = nn.LSTM(dim_combined, dim_output, bidirectional=args.bi)
        self.fc = nn.Linear(dim_output*(2 if args.bi else 1), args.c) # for last hidden state to output classification result
        self.bi = args.bi
        self.c = args.c
        self.no_fc = args.no_fc
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)
        print(self.fc)
    def forward(self, rs, xs):
        # print(f'rs, xs shape = {rs.shape, xs.shape}')
        if self.no_fc:
            hiddens = [self.encoders[i](rs[0 if self.inception_list[i] == False else 1], xs[0 if self.inception_list[i] == False else 1], self.no_fc) for i in range(3)]   #rs[0] or [1] is to control image size for inception model, fuxx you
            # print('hiddens shape', hiddens[0].shape, hiddens[1].shape, hiddens[2].shape, hiddens[3].shape, hiddens[4].shape)
        else:
            hiddens = [self.softmax(self.encoders[i](rs[0 if self.inception_list[i] == False else 1], xs[0 if self.inception_list[i] == False else 1])) for i in range(3)]
            # print('hiddens shape', hiddens[0].shape, hiddens[1].shape, hiddens[2].shape, hiddens[3].shape, hiddens[4].shape)
        hiddens = torch.cat([torch.transpose(hidden.reshape(-1, 50, hidden.size()[-1]), 0, 1) for hidden in hiddens], dim=-1)
        ouptput, (c, h) = self.lstm(self.dropout(hiddens))
        last_h = torch.cat([h[-1], h[-2]], dim=-1) if self.bi else h[-1]
        return self.fc(self.dropout(last_h))

class POOL5(nn.Module):
    def __init__(self, args, paths=None):
        super(POOL5, self).__init__()
        dim_combined = 3840 if args.no_fc else 10#)*(2 if args.z else (args.c+1))
        encoders = []
        for arch, two_layer, ref in [('resnet34', True, 'all0'), ('vgg16', False, 'all1'), ('vgg16', True, 'all1'), ('inception', False, 'all0'), ('inception', True, 'mean')]:
            encoders.append(RefCLS(arch, False, args.combine, two_layer, ref))
        self.encoders = nn.ModuleList(encoders)
        if paths:
#            print('load pretrained CNN')
            for i in range(5):
#                print(i)
                self.encoders[i].load_state_dict(torch.load(paths[i]))
        dim_output = args.dim_lstm
        self.mpool = nn.AdaptiveMaxPool1d(1)
        self.apool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dim_combined*(2 if args.pool=='ma' else 1), args.c)
        self.c = args.c
        self.no_fc = args.no_fc
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = args.pool
    def forward(self, rs, xs):
        print(f'rs, xs shape = {rs.shape, xs.shape}')
        if self.no_fc:
            hiddens = [self.encoders[i](rs[0 if i<3 else 1], xs[0 if i<3 else 1], self.no_fc) for i in range(5)]
            # print('hiddens shape', hiddens[0].shape, hiddens[1].shape, hiddens[2].shape, hiddens[3].shape, hiddens[4].shape)
        else:
            hiddens = [self.softmax(self.encoders[i](rs[0 if i<3 else 1], xs[0 if i<3 else 1])) for i in range(5)]
            # print('hiddens shape', hiddens[0].shape, hiddens[1].shape, hiddens[2].shape, hiddens[3].shape, hiddens[4].shape)
        hiddens = torch.cat([torch.transpose(hidden.reshape(-1, 50, hidden.size()[-1]), 0, 1) for hidden in hiddens], dim=-1)
        hiddens = hiddens.movedim(0, 2)
        pooled1 = self.mpool(hiddens).squeeze(2)
        pooled2 = self.apool(hiddens).squeeze(2)
        if self.pool=='ma':
            return self.fc(self.dropout(torch.cat([pooled1, pooled2], dim=-1)))
        else:
            return self.fc(self.dropout(pooled1 if self.pool=='m' else pooled2))

class CLS(nn.Module):
    def __init__(self, arch, pretrained, combine, two_layer, args, c=-1):
        super(CLS, self).__init__()
        self.encoder = Encoder(arch=arch, pretrained=True) if arch!='inception' else InceptionResnetV1(pretrained=('vggface2'))# if pretrained else None))
#        self.encoder.load_state_dict(torch.load(f'{arch}_encoder0.ckpt'))
        self.c = 3
        self.z = arch=='resnet34'#args.z
        print(arch, self.z)
        dim_combined = args.emb_dim*(2 if self.z else (self.c+1))#dim_embedding#4*dim_embedding if combine=='cat' else dim_embedding
        self.cls = nn.Linear(dim_combined, self.c) if not two_layer else nn.Sequential(nn.Linear(dim_combined, 256), nn.ReLU(), nn.Dropout(), nn.Linear(256, 1))
        self.combine = combine
        self.normal = args.normal
        self.crop = args.crop
    def forward(self, r, x, no_fc=False):
        r = self.encoder(r)
        r = r.reshape(-1 , self.c if not self.z else 1, r.size()[-1]).mean(dim=0)
        x = self.encoder(x)
        r = r.reshape(1, -1).repeat((len(x), 1))
        if no_fc:
            return torch.cat((r, x), dim=-1)
        preds = self.cls(torch.cat((r, x), dim=-1))
        return preds.reshape(-1, self.c)

class RefCLS(nn.Module):
    def __init__(self, arch, pretrained, combine, two_layer, reference, emb_dim=512, c=3):
        super(RefCLS, self).__init__() # seems that pretrained argument doesn't change any program behavior
        self.encoder = Encoder(arch=arch, pretrained=True) if arch!='inception' else InceptionResnetV1(pretrained='vggface2')
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
        if no_fc:
            return embs
        preds = self.cls(embs)
        return preds.reshape(-1, self.c)#, embs

