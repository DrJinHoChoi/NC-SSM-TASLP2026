#!/usr/bin/env python3
"""NC-Conv Lane Detection on CULane.
Upload to Colab with: !git pull && python culane_ncconv.py --culane_root /content/CULane
Or run locally: python culane_ncconv.py --culane_root C:/Users/jinho/Downloads/CULane/CULane
"""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

# =====================================================================
# NC-Conv Block
# =====================================================================
class NCConvBlock(nn.Module):
    def __init__(self, ch, ks=3):
        super().__init__()
        self.static_dw = nn.Conv2d(ch,ch,ks,padding=ks//2,groups=ch,bias=False)
        self.static_bn = nn.BatchNorm2d(ch)
        self.dynamic_dw = nn.Conv2d(ch,ch,ks,padding=ks//2,groups=ch,bias=False)
        self.dynamic_bn = nn.BatchNorm2d(ch)
        self.dyn_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(ch,ch),nn.Sigmoid())
        self.sigma_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),
            nn.Linear(ch,ch//4),nn.SiLU(),nn.Linear(ch//4,1))
        self.pw = nn.Sequential(nn.Conv2d(ch,ch,1,bias=False),nn.BatchNorm2d(ch),nn.SiLU())
    def forward(self, x):
        sigma = torch.sigmoid(self.sigma_net(x)).unsqueeze(-1).unsqueeze(-1)
        h_s = self.static_bn(self.static_dw(x))
        h_d = self.dynamic_bn(self.dynamic_dw(x))
        h_d = h_d * self.dyn_gate(x).unsqueeze(-1).unsqueeze(-1)
        return x + self.pw(F.silu(sigma * h_d + (1 - sigma) * h_s))

# =====================================================================
# Lane Detection Models
# =====================================================================
class LaneHead(nn.Module):
    """Row-anchor lane detection head."""
    def __init__(self, in_ch, n_lanes=4, n_anchors=18):
        super().__init__()
        self.n_lanes = n_lanes
        self.n_anchors = n_anchors
        d_mid = max(in_ch * 2, 64)
        self.row_fc = nn.Linear(in_ch, d_mid)
        self.lane_fc = nn.Linear(d_mid, n_lanes * 2)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        B, C, H, W = x.shape
        # Pool to n_anchors rows
        x = x.mean(dim=3)  # (B, C, H) avg over width
        x = F.adaptive_avg_pool1d(x, self.n_anchors).transpose(1, 2)  # (B, n_anchors, C)
        x = F.silu(self.row_fc(x))
        x = self.dropout(x)
        preds = self.lane_fc(x)  # (B, n_anchors, n_lanes*2)
        return preds.view(B, self.n_anchors, self.n_lanes, 2).permute(0, 2, 1, 3)

class StandardLaneNet(nn.Module):
    def __init__(self, c1=48, c2=96, c3=192, n_lanes=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,c1,3,padding=1,bias=False),nn.BatchNorm2d(c1),nn.SiLU(),
            *[nn.Sequential(
                nn.Conv2d(c1,c1,3,padding=1,groups=c1,bias=False),nn.BatchNorm2d(c1),nn.SiLU(),
                nn.Conv2d(c1,c1,1,bias=False),nn.BatchNorm2d(c1),nn.SiLU()) for _ in range(3)],
            nn.Conv2d(c1,c2,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(c2),nn.SiLU(),
            *[nn.Sequential(
                nn.Conv2d(c2,c2,3,padding=1,groups=c2,bias=False),nn.BatchNorm2d(c2),nn.SiLU(),
                nn.Conv2d(c2,c2,1,bias=False),nn.BatchNorm2d(c2),nn.SiLU()) for _ in range(3)],
            nn.Conv2d(c2,c3,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(c3),nn.SiLU())
        self.head = LaneHead(c3, n_lanes)
    def forward(self, x):
        return self.head(self.backbone(x))

class NCConvLaneNet(nn.Module):
    def __init__(self, c1=44, c2=88, c3=176, n_lanes=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,c1,3,padding=1,bias=False),nn.BatchNorm2d(c1),nn.SiLU())
        self.s1 = nn.Sequential(*[NCConvBlock(c1) for _ in range(3)])
        self.down1 = nn.Sequential(nn.Conv2d(c1,c2,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(c2),nn.SiLU())
        self.s2 = nn.Sequential(*[NCConvBlock(c2) for _ in range(3)])
        self.down2 = nn.Sequential(nn.Conv2d(c2,c3,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(c3),nn.SiLU())
        self.head = LaneHead(c3, n_lanes)
    def forward(self, x):
        return self.head(self.down2(self.s2(self.down1(self.s1(self.stem(x))))))

# =====================================================================
# CULane Dataset
# =====================================================================
class CULaneDataset(Dataset):
    """CULane lane detection dataset.

    Conditions in test_split:
      test0_normal, test2_hlight (glare), test3_shadow, test8_night
    """
    def __init__(self, root, split='train', img_size=(288, 512), n_lanes=4, n_anchors=18):
        self.root = root
        self.img_size = img_size  # (H, W)
        self.n_lanes = n_lanes
        self.n_anchors = n_anchors

        if split == 'train':
            list_file = os.path.join(root, 'list', 'train_gt.txt')
        elif split == 'val':
            list_file = os.path.join(root, 'list', 'val.txt')
        else:
            list_file = os.path.join(root, 'list', 'test.txt')

        self.data = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    img_path = parts[0]
                    self.data.append(img_path)

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Anchor y positions (normalized)
        self.anchor_ys = torch.linspace(0.4, 1.0, n_anchors)

    def __len__(self):
        return len(self.data)

    def _load_annotation(self, img_path):
        """Load lane annotation from .lines.txt file."""
        ann_path = os.path.join(self.root, img_path.replace('.jpg', '.lines.txt').lstrip('/'))
        gt_x = torch.zeros(self.n_lanes, self.n_anchors)
        gt_conf = torch.zeros(self.n_lanes, self.n_anchors)

        if not os.path.exists(ann_path):
            return gt_x, gt_conf

        try:
            lanes = []
            with open(ann_path, 'r') as f:
                for line in f:
                    pts = list(map(float, line.strip().split()))
                    if len(pts) >= 4:
                        xs = pts[0::2]
                        ys = pts[1::2]
                        lanes.append(list(zip(xs, ys)))

            for li, lane in enumerate(lanes[:self.n_lanes]):
                if len(lane) < 2:
                    continue
                lane_xs = [p[0] for p in lane]
                lane_ys = [p[1] for p in lane]

                # Interpolate to anchor positions
                for ai, ay in enumerate(self.anchor_ys):
                    y_target = ay.item() * 590  # original image height
                    # Find closest points
                    for j in range(len(lane_ys) - 1):
                        if (lane_ys[j] <= y_target <= lane_ys[j+1]) or \
                           (lane_ys[j] >= y_target >= lane_ys[j+1]):
                            t = (y_target - lane_ys[j]) / (lane_ys[j+1] - lane_ys[j] + 1e-6)
                            x_interp = lane_xs[j] + t * (lane_xs[j+1] - lane_xs[j])
                            gt_x[li, ai] = x_interp / 1640  # normalize
                            gt_conf[li, ai] = 1.0
                            break
        except:
            pass

        return gt_x, gt_conf

    def __getitem__(self, idx):
        img_path = self.data[idx]
        full_path = os.path.join(self.root, img_path.lstrip('/'))

        try:
            img = Image.open(full_path).convert('RGB')
        except:
            img = Image.new('RGB', (1640, 590))

        img = self.transform(img)
        gt_x, gt_conf = self._load_annotation(img_path)

        return img, gt_x, gt_conf

class CULaneConditionDataset(Dataset):
    """CULane test split by condition (normal/night/shadow/hlight)."""
    def __init__(self, root, condition='normal', img_size=(288, 512), n_lanes=4, n_anchors=18):
        self.root = root
        self.img_size = img_size
        self.n_lanes = n_lanes
        self.n_anchors = n_anchors

        cond_map = {
            'normal': 'test0_normal.txt',
            'crowd': 'test1_crowd.txt',
            'hlight': 'test2_hlight.txt',
            'shadow': 'test3_shadow.txt',
            'noline': 'test4_noline.txt',
            'arrow': 'test5_arrow.txt',
            'curve': 'test6_curve.txt',
            'cross': 'test7_cross.txt',
            'night': 'test8_night.txt',
        }

        list_file = os.path.join(root, 'list', 'test_split', cond_map[condition])
        self.data = []
        with open(list_file, 'r') as f:
            for line in f:
                self.data.append(line.strip().split()[0])

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.anchor_ys = torch.linspace(0.4, 1.0, n_anchors)

    def __len__(self):
        return len(self.data)

    def _load_annotation(self, img_path):
        ann_path = os.path.join(self.root, img_path.replace('.jpg', '.lines.txt').lstrip('/'))
        gt_x = torch.zeros(self.n_lanes, self.n_anchors)
        gt_conf = torch.zeros(self.n_lanes, self.n_anchors)
        if not os.path.exists(ann_path):
            return gt_x, gt_conf
        try:
            lanes = []
            with open(ann_path, 'r') as f:
                for line in f:
                    pts = list(map(float, line.strip().split()))
                    if len(pts) >= 4:
                        lanes.append(list(zip(pts[0::2], pts[1::2])))
            for li, lane in enumerate(lanes[:self.n_lanes]):
                if len(lane) < 2: continue
                xs = [p[0] for p in lane]; ys = [p[1] for p in lane]
                for ai, ay in enumerate(self.anchor_ys):
                    yt = ay.item() * 590
                    for j in range(len(ys)-1):
                        if (ys[j]<=yt<=ys[j+1]) or (ys[j]>=yt>=ys[j+1]):
                            t = (yt-ys[j])/(ys[j+1]-ys[j]+1e-6)
                            gt_x[li,ai] = (xs[j]+t*(xs[j+1]-xs[j]))/1640
                            gt_conf[li,ai] = 1.0; break
        except: pass
        return gt_x, gt_conf

    def __getitem__(self, idx):
        img_path = self.data[idx]
        full_path = os.path.join(self.root, img_path.lstrip('/'))
        try: img = Image.open(full_path).convert('RGB')
        except: img = Image.new('RGB', (1640, 590))
        img = self.transform(img)
        gt_x, gt_conf = self._load_annotation(img_path)
        return img, gt_x, gt_conf

# =====================================================================
# Loss
# =====================================================================
class LaneLoss(nn.Module):
    def __init__(self, sim_weight=1.0):
        super().__init__()
        self.sim_weight = sim_weight
    def forward(self, preds, gt_x, gt_conf):
        pred_x = torch.sigmoid(preds[..., 0])
        pred_conf = preds[..., 1]
        mask = gt_conf > 0.5
        n_pos = mask.sum().clamp(min=1)
        pos_loss = (F.l1_loss(pred_x, gt_x, reduction='none') * mask).sum() / n_pos
        conf_loss = F.binary_cross_entropy_with_logits(pred_conf, gt_conf)
        if pred_x.size(2) > 1:
            diff = (pred_x[:,:,1:] - pred_x[:,:,:-1]).pow(2)
            sim_loss = (diff * mask[:,:,1:]).sum() / mask[:,:,1:].sum().clamp(1)
        else:
            sim_loss = 0
        return pos_loss + conf_loss + self.sim_weight * sim_loss

# =====================================================================
# Training + Evaluation
# =====================================================================
def train_and_eval(culane_root, device='cuda'):
    IMG_SIZE = (288, 512)
    EPOCHS = 50
    BATCH = 32
    LR = 1e-3

    print('Loading CULane...')
    train_ds = CULaneDataset(culane_root, 'train', IMG_SIZE)
    val_ds = CULaneDataset(culane_root, 'val', IMG_SIZE)
    print(f'  Train: {len(train_ds)} | Val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=4, pin_memory=True)

    criterion = LaneLoss()
    results = {}

    for name, model in [
        ('Std CNN', StandardLaneNet(48, 96, 192)),
        ('NC-Conv', NCConvLaneNet(44, 88, 176)),
    ]:
        model = model.to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f'\n{"="*60}')
        print(f'  {name}: {params:,} params')
        print(f'{"="*60}')

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

        best_acc = 0; best_state = None
        for ep in range(1, EPOCHS+1):
            t0 = time.time()
            model.train(); total_loss = 0; nb = 0
            for imgs, gt_x, gt_conf in train_loader:
                imgs, gt_x, gt_conf = imgs.to(device), gt_x.to(device), gt_conf.to(device)
                opt.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, gt_x, gt_conf)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item(); nb += 1

            model.eval()
            correct, total_pts = 0, 0
            with torch.no_grad():
                for imgs, gt_x, gt_conf in val_loader:
                    imgs, gt_x, gt_conf = imgs.to(device), gt_x.to(device), gt_conf.to(device)
                    preds = model(imgs)
                    pred_x = torch.sigmoid(preds[..., 0])
                    pred_c = torch.sigmoid(preds[..., 1])
                    mask = gt_conf > 0.5
                    correct += ((pred_x - gt_x).abs() < 0.05).logical_and(pred_c > 0.3).logical_and(mask).sum().item()
                    total_pts += mask.sum().item()

            acc = correct / max(total_pts, 1) * 100
            sched.step()
            if acc > best_acc: best_acc = acc; best_state = {k:v.clone() for k,v in model.state_dict().items()}
            if ep % 10 == 0 or ep <= 3 or acc >= best_acc:
                print(f'  E{ep:>2}/{EPOCHS} | loss={total_loss/nb:.4f} | acc={acc:.1f}% | {time.time()-t0:.0f}s{"*" if acc>=best_acc else ""}')

        model.load_state_dict(best_state); model.eval()

        # Per-condition evaluation
        conditions = ['normal', 'hlight', 'shadow', 'night']
        cond_results = {}
        for cond in conditions:
            try:
                cds = CULaneConditionDataset(culane_root, cond, IMG_SIZE)
                cdl = DataLoader(cds, batch_size=BATCH, num_workers=4)
                c, t = 0, 0
                with torch.no_grad():
                    for imgs, gt_x, gt_conf in cdl:
                        imgs, gt_x, gt_conf = imgs.to(device), gt_x.to(device), gt_conf.to(device)
                        preds = model(imgs)
                        pred_x = torch.sigmoid(preds[..., 0])
                        pred_c = torch.sigmoid(preds[..., 1])
                        mask = gt_conf > 0.5
                        c += ((pred_x-gt_x).abs()<0.05).logical_and(pred_c>0.3).logical_and(mask).sum().item()
                        t += mask.sum().item()
                cond_results[cond] = c / max(t, 1) * 100
                print(f'    {cond:<10}: {cond_results[cond]:.1f}% ({len(cds)} images)')
            except Exception as e:
                print(f'    {cond:<10}: ERROR - {e}')
                cond_results[cond] = 0

        results[name] = {'val_acc': best_acc, 'conditions': cond_results, 'params': params}
        del opt, sched; torch.cuda.empty_cache()

    # Final table
    print(f'\n{"="*70}')
    print(f'  CULane Lane Detection: NC-Conv vs Standard CNN')
    print(f'{"="*70}')
    conds = ['normal', 'hlight', 'shadow', 'night']
    audio = {'normal':'Clean','hlight':'Glare (non-stat)','shadow':'Shadow (partial)','night':'Night (stationary)'}
    print(f'  {"Condition":<12} | {"Audio":<20} | {"Std CNN":>8} | {"NC-Conv":>8} | {"Gap":>6}')
    print(f'  {"-"*65}')
    for c in conds:
        s = results['Std CNN']['conditions'].get(c, 0)
        n = results['NC-Conv']['conditions'].get(c, 0)
        print(f'  {c:<12} | {audio[c]:<20} | {s:>7.1f}% | {n:>7.1f}% | {n-s:>+5.1f}%')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--culane_root', default='C:/Users/jinho/Downloads/CULane/CULane')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train_and_eval(args.culane_root, args.device)
