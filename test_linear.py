import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dinov2.models.vision_transformer import vit_base, vit_large
from collections import OrderedDict


def get_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def load_dinov2_vitb(checkpoint_path, device):


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
    return model


def extract_features(model, dataloader, device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, target in tqdm(dataloader, desc='Extracting Features'):
            images = images.to(device)
            output = model.forward_features(images)["x_norm_clstoken"]
            features.append(output.cpu())
            labels.append(target)
    return torch.cat(features), torch.cat(labels)


def test_linear_logits(model, test_loader, linear_clf, device):
    model.eval()
    linear_clf.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)["x_norm_clstoken"]
            logits = linear_clf(feats)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    base_path = "DATASET PATH"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "MODEL PATH"
    model = load_dinov2_vitb(model_path, device)
    results = []

    for task_name in sorted(os.listdir(base_path)):
        task_path = os.path.join(base_path, task_name)
        if not os.path.isdir(task_path):
            continue
        print(f"Evaluating task {task_name}...")

        dataset = datasets.ImageFolder(task_path, transform=get_transforms())
        if len(dataset.classes) < 2:
            print(f"Skipping task {task_name} since less than 2 classes")
            continue

        all_labels = np.array([label for _, label in dataset])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_accuracies, fold_f1s, fold_aurocs = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
            print(f"  Fold {fold_idx + 1}/5")

            train_set = Subset(dataset, train_idx)
            test_set = Subset(dataset, test_idx)

            train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

            train_feats, train_labels = extract_features(model, train_loader, device)
            linear_clf = LinearClassifier(train_feats.shape[1], len(dataset.classes)).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(linear_clf.parameters(), lr=1e-3)

            linear_clf.train()
            for epoch in range(5):
                perm = torch.randperm(train_feats.size(0))
                for i in range(0, train_feats.size(0), 64):
                    idx = perm[i:i + 64]
                    feats_batch = train_feats[idx].to(device)
                    labels_batch = train_labels[idx].to(device)
                    optimizer.zero_grad()
                    outputs = linear_clf(feats_batch)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()

            logits, true_labels = test_linear_logits(model, test_loader, linear_clf, device)
            pred_labels = logits.argmax(dim=1)

            acc = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average='weighted')

            if len(dataset.classes) == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
                auroc = roc_auc_score(true_labels, probs)
            else:
                probs = torch.softmax(logits, dim=1)
                try:
                    auroc = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
                except ValueError:
                    auroc = float('nan')  # e.g., if one class not in y_true

            print(f"    Fold {fold_idx + 1} Acc: {acc:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")

            fold_accuracies.append(acc)
            fold_f1s.append(f1) 
            fold_aurocs.append(auroc)

            result_entry = {'task': task_name}
            for i in range(len(fold_accuracies)):
                result_entry[f'Fold{i}_Acc'] = round(fold_accuracies[i], 4)
                result_entry[f'Fold{i}_F1'] = round(fold_f1s[i], 4)
                result_entry[f'Fold{i}_AUROC'] = round(fold_aurocs[i], 4) if not np.isnan(fold_aurocs[i]) else None
            
            result_entry.update({
                'Mean_Acc': round(np.mean(fold_accuracies), 4),
                'Mean_F1': round(np.mean(fold_f1s), 4),
                'Mean_AUROC': round(np.nanmean(fold_aurocs), 4),
                'num_classes': len(dataset.classes)
            })
 
        results.append(result_entry)

    df = pd.DataFrame(results)
    save_path = "OUTPUT PATH"
    df.to_excel(save_path, index=False)
    print(f"Saved results to {save_path}")


if __name__ == '__main__':
    main()
