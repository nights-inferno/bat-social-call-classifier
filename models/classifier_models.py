
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin

from models.focal_loss import FocalLoss


class BalancedMLP(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(
        self,
        input_dim=1536,
        hidden_dim=128, 
        lr=0.001, 
        epochs=50, 
        dropout=0.2,
        balanced = False, 
        focal_loss : bool = False,
        focal_gamma : float = 2.0,
        focal_alpha : float = 0.25,
        batch_norm : bool = False,
        batch_size=32,
        random_state = None
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.model = None
        self.classes_ = None
        self.balanced : bool = balanced
        self.focal_loss : bool = focal_loss
        self.focal_gamma : float = focal_gamma
        self.focal_alpha : float = focal_alpha
        self.batch_norm : bool = batch_norm
        self.batch_size = batch_size
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None
        self.train_loss_history_ = []
        self.val_loss_history_ = []

    def _build_model(self, output_dim):
        return nn.Sequential(
            nn.BatchNorm1d(self.input_dim) if self.batch_norm else nn.Identity(),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def fit(self, X, y,X_val = None,y_val = None,**kwargs):
        #random state initialisation 
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
            # Create a dedicated seeded generator for DataLoader shuffling
            g = torch.Generator()
            g.manual_seed(self.random_state)
        else:
            g = None

        # Handle scikit-learn prefix kwargs routing (e.g., model__X_val) safely
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if X_val is None and 'model__X_val' in kwargs:
            X_val = kwargs['model__X_val']
        if y_val is None and 'model__y_val' in kwargs:
            y_val = kwargs['model__y_val']

        # Reset history tracking on fresh calls
        self.train_loss_history_ = []
        self.val_loss_history_ = []

        # Convert to Tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
        
        # 1. Handle Class Imbalance Automatically
        # pos_weight = (count_negative / count_positive)
        num_pos = y_tensor.sum(dim=0)
        num_neg = y_tensor.size(0) - num_pos
        # Add small epsilon to avoid division by zero
        pos_weight = num_neg / (num_pos + 1e-6) 
        
        # 2. Setup Training
        self.model_ = self._build_model(y.shape[1]).to(device)
        if self.focal_loss:
            criterion = FocalLoss(gamma=self.focal_gamma, alpha=self.focal_alpha, task_type='multi-label')
        else :
            pos_weight_dev = pos_weight.to(device) if self.balanced else None
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_dev if self.balanced else None)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        # Convert to dataset iterator to keep batch normalizations mathematically stable
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(X)), shuffle=True,generator = g)

        # 3. Training Loop
        for epoch in range(self.epochs):
            self.model_.train()
            epoch_train_loss = 0.0
            batch_count = 0

            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                batch_count += 1
            #Average training loss for epoch    
            self.train_loss_history_.append(epoch_train_loss / batch_count)
            #Compute validation
            if has_validation:
                self.model_.eval()
                with torch.no_grad():
                    X_v, y_v = X_val_tensor.to(device), y_val_tensor.to(device)
                    val_outputs = self.model_(X_v)
                    val_loss = criterion(val_outputs, y_v)
                    self.val_loss_history_.append(val_loss.item())
            else:
                # Fallback step so dimensions align nicely across plots
                self.val_loss_history_.append(np.nan)
            
        self.classes_ = np.arange(y.shape[1])
        return self

    def predict_proba(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            logits = self.model_(X_tensor)
            # BCEWithLogitsLoss outputs logits; sigmoid turns them into 0-1 probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)  
    

class MultilabelPrevalenceBaseline(BaseEstimator,ClassifierMixin):
    def __init__(self, type : str = 'stochastic'):
        self.type : str = type
        
    def fit(self, X, y):
        self.prevalence_ = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_labels = len(self.prevalence_)
        if self.type == 'stochastic':
            probs = np.zeros((n_samples, n_labels))
            for j in range(n_labels):
                alpha = self.prevalence_[j] * 10 + 1e-3
                beta = (1 - self.prevalence_[j]) * 10 + 1e-3
                probs[:, j] = np.random.beta(
                    alpha,
                    beta,
                    size=n_samples
                )
        elif self.type == 'majority' :
            pred = np.zeros(n_labels)
            pred[np.argmax(self.prevalence_)] = 1
            probs = np.tile(
                pred,
                (n_samples, 1)
            )
        elif self.type == 'prevalence' :
            probs = np.tile(
                self.prevalence_,
                (n_samples, 1)
            )
        else :
            raise ValueError(f"Unknown type {self.type} for MultilabelPrevalenceBaseline")

        return probs

    def predict(self, X,threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
