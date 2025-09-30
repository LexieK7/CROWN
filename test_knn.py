import os
import torch
import openpyxl
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from dinov2.models.vision_transformer import vit_base, vit_large


def extract_features(task_path, model, transform, device):
    dataset = datasets.ImageFolder(root=task_path, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"Extracting {os.path.basename(task_path)}"):
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)["x_norm_clstoken"]
            features.append(feats.cpu())
            labels.append(lbls)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def knn_predict(test_feats, train_feats, train_labels, k, temperature):
 
    sim = torch.mm(test_feats, train_feats.T) / temperature
    topk_sim, topk_indices = sim.topk(k=k, dim=1, largest=True, sorted=True)
    topk_labels = train_labels[topk_indices]  # (n_test, k)

    # One-hot votes weighted by similarity
    num_classes = train_labels.max().item() + 1
    votes = torch.zeros(test_feats.size(0), num_classes, device=test_feats.device)
    for i in range(k):
        votes.scatter_add_(1, topk_labels[:, i:i+1], topk_sim[:, i:i+1])
    return votes


def run_knn_5fold(features, labels, k=20, temperature=0.07):
    indices = torch.randperm(len(labels))
    folds = torch.chunk(indices, 5)
    accs = []

    for i in range(5):
        test_idx = folds[i]
        train_idx = torch.cat([folds[j] for j in range(5) if j != i])

        train_feats, train_labels = features[train_idx], labels[train_idx]
        test_feats, test_labels = features[test_idx], labels[test_idx]

        pred_scores = knn_predict(test_feats, train_feats, train_labels, k, temperature)
        preds = pred_scores.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100
        accs.append(acc)
    return accs


def main():
    checkpoint_path = "MODEL PATH"
    tasks_root = "DATASET PATH"
    excel_path = "OUTPUT PATH"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = vit_large(
        patch_size=16,
        img_size=224, 
        init_values=1.0,
        block_chunks=4,
        ffn_layer="swiglufused",
    )
    

    state_dict = torch.load(checkpoint_path, map_location="cpu")["teacher"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_key = k[len("backbone."):]
            new_state_dict[new_key] = v
    filtered_state_dict = {
        k: v for k, v in new_state_dict.items()
        if not k.startswith("dino_head") and not k.startswith("ibot_head")
    }
    model.load_state_dict(filtered_state_dict, strict=True)
    model.to(device).eval()
    
    

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["TaskID", "Fold0", "Fold1", "Fold2", "Fold3", "Fold4", "Mean"])

    task_ids = sorted([d for d in os.listdir(tasks_root) if os.path.isdir(os.path.join(tasks_root, d))])
    for task_id in tqdm(task_ids):
        task_path = os.path.join(tasks_root, task_id)
        try:
            features, labels = extract_features(task_path, model, transform, device)
            accs = run_knn_5fold(features, labels, k=20, temperature=0.07)
            mean_acc = np.mean(accs)
            ws.append([task_id] + [round(a, 4) for a in accs] + [round(mean_acc, 4)])
        except Exception as e:
            print(f"[Error] Task {task_id} failed: {e}")
            ws.append([task_id] + ["Failed"] * 6)

    wb.save(excel_path)
    print(f"Saved results to {excel_path}")


if __name__ == "__main__":
    main()

