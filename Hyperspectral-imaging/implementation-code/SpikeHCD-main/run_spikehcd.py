import argparse
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from src.utils import load_santa_barbara_pairs, get_coordinates_labels, get_train_test, Grammar
from src.model import QKFormer

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data from {args.data_path}...")
    x1, x2, y = load_santa_barbara_pairs(args.data_path)
    
    print("Preparing patches...")
    coords, labels = get_coordinates_labels(y)
    (tr_c, tr_y), (val_c, val_y), (te_c, te_y) = get_train_test(coords, labels)
    
    tr_x1, tr_x2 = Grammar(x1, x2, tr_c)
    val_x1, val_x2 = Grammar(x1, x2, val_c)
    te_x1, te_x2 = Grammar(x1, x2, te_c)
    
    model = QKFormer(in_channels=x1.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        # Simple batch loop
        perm = torch.randperm(len(tr_y))
        epoch_loss = 0
        for i in range(0, len(tr_y), args.batch_size):
            idx = perm[i:i+args.batch_size]
            bx1 = torch.tensor(tr_x1[idx], dtype=torch.float32).to(device)
            bx2 = torch.tensor(tr_x2[idx], dtype=torch.float32).to(device)
            by = torch.tensor(tr_y[idx], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            out = model(bx1, bx2)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f}")

    # Evaluation
    print("Evaluating...")
    model.eval()
    with torch.no_grad():
        tx1 = torch.tensor(te_x1, dtype=torch.float32).to(device)
        tx2 = torch.tensor(te_x2, dtype=torch.float32).to(device)
        preds = torch.argmax(model(tx1, tx2), dim=1).cpu().numpy()
    
    print(f"OA: {accuracy_score(te_y, preds):.4f}")
    print(f"Kappa: {cohen_kappa_score(te_y, preds):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4e-4)
    args = parser.parse_args()
    train(args)
