import numpy as np
import scipy.io as sio
import os

def load_santa_barbara_pairs(root_dir, year1="T1", year2="T2"):
    try:
        d1 = sio.loadmat(os.path.join(root_dir, f"farmland_{year1}.mat"))
        d2 = sio.loadmat(os.path.join(root_dir, f"farmland_{year2}.mat"))
        gt = sio.loadmat(os.path.join(root_dir, "farmland_binary.mat"))
        x1 = d1["T1"].astype(np.float32)
        x2 = d2["T2"].astype(np.float32)
        y_raw = gt["Binary"].squeeze()
        # Process Labels: 1->Changed(1), 2->Unchanged(0)
        y = np.zeros_like(y_raw, dtype=np.int64)
        y[y_raw == 1] = 1; y[y_raw == 2] = 0
        # Normalize
        x1 = (x1 - x1.min()) / (x1.max() - x1.min() + 1e-8)
        x2 = (x2 - x2.min()) / (x2.max() - x2.min() + 1e-8)
        return x1, x2, y
    except Exception as e:
        print(f"Error loading data: {e}. Returning dummy data.")
        return np.random.randn(100,100,198).astype(np.float32), np.random.randn(100,100,198).astype(np.float32), np.random.randint(0,2,(100,100))

def get_coordinates_labels(y_hsi):
    row_coords, col_coords, labels = [], [], []
    for lbl in np.unique(y_hsi):
        locs = np.where(y_hsi == lbl)
        if len(locs[0]) == 0: continue
        row_coords.append(locs[0]); col_coords.append(locs[1])
        labels.append(np.array([lbl]*len(locs[0])))
    if not row_coords: return np.array([]), np.array([])
    return np.stack([np.concatenate(row_coords), np.concatenate(col_coords)], axis=1), np.concatenate(labels)

def get_train_test(coords, labels, val_size=0.1, test_size=0.8):
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        np.random.shuffle(c_idx)
        n, n_test, n_val = len(c_idx), int(len(c_idx)*test_size), int(len(c_idx)*val_size)
        test_idx.extend(c_idx[:n_test]); val_idx.extend(c_idx[n_test:n_test+n_val]); train_idx.extend(c_idx[n_test+n_val:])
    return (coords[train_idx], labels[train_idx]), (coords[val_idx], labels[val_idx]), (coords[test_idx], labels[test_idx])

def Grammar(img1, img2, coords, size=3):
    margin = (size - 1) // 2
    pad = lambda x: np.pad(x, [(margin, margin), (margin, margin), (0,0)], mode="constant")
    i1, i2 = pad(img1), pad(img2)
    p1 = [i1[r:r+size, c:c+size] for r, c in coords]
    p2 = [i2[r:r+size, c:c+size] for r, c in coords]
    return np.array(p1).reshape(len(p1), -1, img1.shape[2]), np.array(p2).reshape(len(p2), -1, img2.shape[2])
