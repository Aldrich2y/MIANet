import torch
from torch import nn
import torch.nn.functional as F
import model.vgg as vgg_models
from util.util import load_obj
from model.PSPNet import OneModel as PSPNet
import pickle

# load word embeddings
def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        embed = torch.from_numpy(pickle.load(f, encoding="latin-1"))
    embed.requires_grad = False
    return embed   # [21, 300]

# The Implementation of General Information Generator
class GIG(nn.Module):
    def __init__(self, in_channels=556, out_channels=256, hidden_size=256):
        super(GIG, self).__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p=0.2))
            return layers

        if hidden_size:
            self.model = nn.Sequential(
                *block(in_channels, hidden_size),
                nn.Linear(hidden_size, out_channels),
                nn.Dropout(0.3)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_channels, out_channels),)
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, embeddings):
        return self.model(embeddings)

# extract the instance support prototype
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


# The Implementation of Hierarchical Prior Module
def HPM(main, aux, mask, bins):
    """
    :param main: query features
    :param aux: compute logits from support features
    :param mask: support mask
    :param bins: [60, 30, 15, 8]
    :return: the cosine similiarity between main and aux
     """
    res = []
    if main.shape[-1] == 30:  # vgg16
        temp_aux = aux
        temp_main = F.interpolate(main, size=(60, 60), mode='bilinear', align_corners=True)
        temp_aux = F.interpolate(temp_aux, size=temp_main.shape[-2:], mode='bilinear', align_corners=True)
        temp_mask = F.interpolate(mask, size=temp_main.shape[-2:], mode='bilinear', align_corners=True)
        temp_aux = temp_aux * temp_mask
        sim_i = cos_similarity(temp_main, temp_aux)
        res.append(sim_i)
        bins=bins[1:]
    for i in range(len(bins)):
        temp_aux = aux
        temp_aux = F.interpolate(temp_aux, size=main.shape[-2:], mode='bilinear', align_corners=True)
        mask = F.interpolate(mask, size=main.shape[-2:], mode='bilinear', align_corners=True)
        temp_aux = temp_aux * mask
        sim_i = cos_similarity(main, temp_aux)
        res.append(sim_i)

        # information channels
        main = main * sim_i
        if i!=len(bins)-1:
            main = F.adaptive_avg_pool2d(main, bins[i+1])
    return res

def cos_similarity(main, aux):
    b, c, h, w = main.shape
    cosine_eps = 1e-7
    main = main.view(b, c, -1).permute(0, 2, 1).contiguous()  # [b, h*w, c]
    main_norm = torch.norm(main, 2, 2, True)
    aux = aux.view(b, c, -1)
    aux_norm = torch.norm(aux, 2, 1, True)

    logits = torch.bmm(main, aux) / (torch.bmm(main_norm, aux_norm) + cosine_eps)  # [b, hw, hw]
    similarity = torch.mean(logits, dim=-1).view(b, h * w)
    similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
            similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
    corr_query = similarity.view(b, 1, h, w)
    return corr_query


class MIANet(nn.Module):
    def __init__(self, args, layers=50, classes=2, zoom_factor=8, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False):
        super(MIANet,self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = args.shot
        self.ppm_scales = ppm_scales
        self.vgg = vgg
        # initial
        self.pspnet = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)


        # load pre-training PSPNet
        weight_path = 'initmodel/{}/split{}/{}/best.pth'.format(args.data_set, args.split,
                                                                                       backbone_str)
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try:
            self.pspnet.load_state_dict(new_param)
            print('INFO: loading PSPNet parameters split: ' + str(args.split))
        except RuntimeError:  # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            self.pspnet.load_state_dict(new_param)
            print('mGPU transfering-------INFO: loading PSPNet parameters split: ' + str(args.split))

        if self.vgg:
            print('INFO: Using VGG_16 bn')

        else:
            print('INFO: Using ResNet {}'.format(layers))

        # using the pre-trained backbone from PSPNet
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = self.pspnet.layer0, self.pspnet.layer1, self.pspnet.layer2, self.pspnet.layer3, self.pspnet.layer4

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.down_features = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        # for FEM structures--------------------
        self.pyramid_bins = self.ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            self.avgpool_list.append(
                nn.AdaptiveAvgPool2d(bin)
            )

        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

        # end FEM structures-------------------------

        # General Information Generator
        self.GIG = GIG(in_channels=300 + reduce_dim, out_channels=reduce_dim, hidden_size=int(reduce_dim/2))

         # The inplementation of Local Feature Generator
        self.LFG = nn.Sequential(
            nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x, s_x=torch.FloatTensor(1, 1, 3, 473, 473).cuda(), s_y=torch.FloatTensor(1, 1, 473, 473).cuda(),
                y=None, class_chosen=[1,2,3,4] ):
        x_size = x.size()
        # class_chosen  [b]
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # print(x.shape,s_x.shape, y.shape, s_y.shape)  torch.Size([4, 3, 473, 473]) torch.Size([4, 1, 3, 473, 473]) torch.Size([4, 473, 473]) torch.Size([4, 1, 473, 473])
        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_features(query_feat)

        #   Support Feature
        mask = (s_y[:, 0, :, :] == 1).float().unsqueeze(1)
        with torch.no_grad():
            supp_feat_0 = self.layer0(s_x[:, 0, :, :, :])
            supp_feat_1 = self.layer1(supp_feat_0)
            supp_feat_2 = self.layer2(supp_feat_1)
            supp_feat_3 = self.layer3(supp_feat_2)
            mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                 align_corners=True)
            supp_feat_4 = self.layer4(supp_feat_3)
            if self.vgg:
                supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                            mode='bilinear', align_corners=True)

        supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat = self.down_features(supp_feat)
        support_out_feat = supp_feat
        # res [b, 21, 1]
        instance_prototype = Weighted_GAP(supp_feat, mask)

        # embeddings
        embeddings = load_obj('embeddings/word2vec_pascal').cuda()   # embeddings [n, d]  coco: [80, 300]
        embeddings = torch.stack([embeddings] * x_size[0], dim=0)    # expand for concat
        bq, cq, hq, wq = query_feat.shape
        instance_prototype = instance_prototype.view(bq, cq).unsqueeze(1).expand(bq, embeddings.shape[1], cq)   # instance prototype
        anchors = self.GIG(torch.cat((embeddings, instance_prototype), dim=-1))  # [b, 21, c + 300]


        # obtain local features
        local_features = self.LFG(support_out_feat)
        if self.training:
            # calculate triplet_loss
            triple_loss = get_triple_loss(anchors, local_features, mask, class_chosen)
        else:
            triple_loss = torch.Tensor([0.0])

        # get corresponding word vectors
        row = list(range(embeddings.shape[0]))
        general_prototype = anchors[row, class_chosen].unsqueeze(-1).unsqueeze(-1)


        corr_query_list = HPM(query_feat_4, supp_feat_4, mask, self.pyramid_bins)

        out_list = []
        pyramid_feat_list = []
        # FEM  output  make predictions
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            bin = tmp_bin
            query_feat_bin = self.avgpool_list[idx](query_feat)
            supp_feat_bin = general_prototype.expand(-1, -1, bin, bin)
            corr_mask_bin = corr_query_list[idx]
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()

            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())
            aux_loss = aux_loss / len(out_list)
            return out.max(1)[1], main_loss, aux_loss, triple_loss   # out, l_seg1, l_seg2, triplet loss
        else:
            return out


def get_triple_loss(anchors, local_features, mask, class_chosen):
    """

    Args:
        anchors: prototypes: [b, n_class, c]
        local_features:  [b, c, h, w]
        mask: [b,1,h,w]
        class_chosen: [b, ]  int

    Returns: triplet_loss

    """
    b, c, h, w = local_features.shape
    mask = F.interpolate(mask.float(), size=(h, w), mode="nearest").long().view(b, -1)  # [b, h*w]
    local_features = local_features.view(b, c, -1).permute(0, 2, 1).contiguous() # [b, h*w, c]
    triplet_loss = torch.Tensor([0.0]).cuda()
    length = 1   # number of triplets

    # hard triplet dig
    count = b
    for i in range(b):
        anchor_list = []
        mask_i = mask[i]  # [h*w]
        negative_list_i = local_features[i][mask_i == 0]
        positive_list_i = local_features[i][mask_i == 1]

        anchor_list_i_mu = anchors[i][int(class_chosen[i])].unsqueeze(0)
        for sample_i in range(length):
            anchor_list.append(anchor_list_i_mu)
        anchor_list = torch.cat(anchor_list, dim=0)

        if positive_list_i.shape[0] <length or negative_list_i.shape[0]< length:  # if none postive or negtive is not found due to down-sapmling
            temp_loss = torch.Tensor([0.0]).cuda()
            count = count - 1
        else:
            temp_loss = hard_triplet_dig(anchor_list, positive_list_i, negative_list_i)

        triplet_loss = triplet_loss + temp_loss

    return triplet_loss / max(count, 1)


def hard_triplet_dig(anchor, positive, negative):
    """

    Args:
        anchor: [length, c]
        positive: [nums_x, c]
        negative: [nums_y, c]
    Returns: triplet loss
    """
    for i in range(anchor.shape[0]):
        edu_distance_pos = F.pairwise_distance(F.normalize(anchor[i], p=2, dim=-1),
                                                 F.normalize(positive, p=2, dim=-1))
        edu_distance_neg = F.pairwise_distance(F.normalize(anchor[i], p=2, dim=-1), torch.mean(F.normalize(negative, p=2, dim=-1), dim=0, keepdim=True))
        neg_val, _ = edu_distance_neg.sort()
        pos_val, _ = edu_distance_pos.sort()

        triplet_loss = max(0, 0.5 + pos_val[-1] - neg_val[0])   # 0.5
    return triplet_loss


