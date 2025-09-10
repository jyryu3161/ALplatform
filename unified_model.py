import os
import logging
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import joblib
from datetime import datetime

class TabMNetwork(nn.Module):
    def __init__(self, n_features, embed_dim=128, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(n_features, embed_dim)
        self.model = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.embed(x)
        return self.model(x)
    
class UnifiedModel:    
    def __init__(self, auto_tune=False, tuning_method='random', **kwargs):
        self.model_type = 'tabm'
        self.auto_tune = auto_tune
        self.tuning_method = tuning_method
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_fitted = False
        self.best_params_ = None
        self.max_threads = kwargs.get('max_threads', 8)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.dropout = kwargs.get('dropout', 0.2)
        self.weight_decay = kwargs.get('weight_decay', 0) # Add weight_decay
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 256)
        self.mc_dropout_samples = kwargs.get('mc_dropout_samples', 50)
        
        logging.info(f"Unified Model ({self.model_type}) - Auto tune: {self.auto_tune}")

    def fit(self, X, y):
        X = self._convert_to_array(X)
        y = self._convert_to_array(y).flatten()
        
        logging.info(f"Training {self.model_type} with {len(X)} samples")
        
        if self.auto_tune:
            logging.info("🔧 Performing hyperparameter tuning...")
            self._tune_hyperparameters(X, y)
        
        self._fit_tabm(X, y)
        self.is_fitted = True
        return self

    def _tune_hyperparameters(self, X, y):
        self._tune_tabm(X, y)

    def _tune_tabm(self, X, y):
        n_features = X.shape[1]
        
        def objective(trial):
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            dropout = trial.suggest_float('dropout', 0.2, 0.7)  # Increased dropout range
            embed_dim = trial.suggest_categorical('embed_dim', [64, 128, 256])
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True) # Added weight_decay
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            temp_model = TabMNetwork(n_features, embed_dim, dropout).to(self.device)
            optimizer = optim.Adam(temp_model.parameters(), lr=lr, weight_decay=weight_decay) # Apply weight_decay
            criterion = nn.MSELoss()
            
            dataset = TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float().view(-1, 1)
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            temp_epochs = min(50, self.epochs)
            for epoch in range(temp_epochs):
                temp_model.train()
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = temp_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            
            temp_model.eval()
            with torch.no_grad():
                X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
                y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1).to(self.device)
                
                outputs = temp_model(X_val_tensor)
                ss_res = torch.sum((y_val_tensor - outputs) ** 2)
                ss_tot = torch.sum((y_val_tensor - torch.mean(y_val_tensor)) ** 2)
                r2_score = (1 - ss_res / ss_tot).item()
            
            del temp_model, optimizer, dataset, loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return r2_score
        
        try:
            study = optuna.create_study(direction='maximize', 
                                       sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=15, show_progress_bar=False)
            
            self.best_params_ = study.best_params
            self.learning_rate = study.best_params['learning_rate']
            self.dropout = study.best_params['dropout']
            self.weight_decay = study.best_params.get('weight_decay', 0) # Save weight_decay
            
            logging.info(f"✅ TabM tuning completed - Best R²: {study.best_value:.4f}")
            logging.info(f"📋 Best params: {self.best_params_}")
            
        except Exception as e:
            logging.warning(f"⚠️ TabM tuning failed: {str(e)}, using defaults")

    def _fit_tabm(self, X, y):
        n_features = X.shape[1]
        
        embed_dim = self.best_params_.get('embed_dim', 128) if self.best_params_ else 128
        learning_rate = self.learning_rate
        dropout = self.dropout
        
        self.model = TabMNetwork(n_features, embed_dim, dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.weight_decay) # Apply weight_decay
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float().view(-1, 1)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            self.model.train()
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.debug(f"Early stopping at epoch {epoch + 1}")
                break
        
        logging.info(f"🎯 TabM trained for {epoch + 1} epochs")

    def predict(self, X):
        if not self.is_fitted:
            logging.error("Model not fitted yet")
            return np.zeros(X.shape[0]), np.zeros(X.shape[0])
        
        X = self._convert_to_array(X)
        
        return self._predict_tabm(X)

    def _predict_tabm(self, X):
        if self.model is None:
            return np.zeros(X.shape[0]), np.zeros(X.shape[0])

        self.model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            for _ in range(self.mc_dropout_samples):
                outputs = self.model(X_tensor)
                predictions.append(outputs.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        return mean_pred, std_pred

    def score(self, X, y):
        X = self._convert_to_array(X)
        y = self._convert_to_array(y).flatten()
        
        mean_pred, _ = self.predict(X)
        
        ss_res = np.sum((y - mean_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)

    def get_model_info(self):
        return {
            'model_type': self.model_type,
            'auto_tune': self.auto_tune,
            'tuning_method': self.tuning_method,
            'is_fitted': self.is_fitted,
            'best_params': self.best_params_,
            'learning_rate': getattr(self, 'learning_rate', None),
            'dropout': getattr(self, 'dropout', None)
        }

    def _convert_to_array(self, data):
        if isinstance(data, list):
            return np.array(data, dtype=np.float32)
        elif hasattr(data, 'to_numpy'):
            return data.to_numpy(dtype=np.float32)
        elif isinstance(data, np.ndarray):
            return data.astype(np.float32)
        return np.array(data, dtype=np.float32)

    def evaluate_all_metrics(self, X, y):
        X = self._convert_to_array(X)
        y = self._convert_to_array(y).flatten()
        
        mean_pred, _ = self.predict(X)
        
        ss_res = np.sum((y - mean_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)
        
        mae = np.mean(np.abs(y - mean_pred))
        
        mse = np.mean((y - mean_pred) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'n_samples': len(y)
        }

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._save_tabm_model(save_path, timestamp)
        self._save_model_metadata(save_path, timestamp)
        
        logging.info(f"Model saved to {save_path}")

    def _save_tabm_model(self, save_path, timestamp):
        if self.model is None:
            logging.warning("No TabM model to save")
            return
        
        model_file = os.path.join(save_path, f'tabm_model_{timestamp}.pth')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': {
                'n_features': self.model.embed.in_features,
                'embed_dim': self.model.embed.out_features,
                'dropout': self.dropout
            },
            'timestamp': timestamp
        }, model_file)

    def _save_model_metadata(self, save_path, timestamp):
        metadata = {
            'model_info': self.get_model_info(),
            'timestamp': timestamp,
            'training_completed': self.is_fitted,
            'device': getattr(self, 'device', 'cpu')
        }
        
        metadata_file = os.path.join(save_path, 'model_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, load_path):
        if not os.path.exists(load_path):
            raise ValueError(f"Model path does not exist: {load_path}")
        
        metadata_file = os.path.join(load_path, 'model_metadata.json')
        if not os.path.exists(metadata_file):
            raise ValueError(f"Model metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        model_info = metadata['model_info']

        self.model_type = model_info['model_type']
        self.best_params_ = model_info.get('best_params')
        self._load_tabm_model(load_path)
        self.is_fitted = True

        logging.info(f"Model loaded from {load_path}")

    def _load_tabm_model(self, load_path):
        model_files = [f for f in os.listdir(load_path) if f.startswith('tabm_model_') and f.endswith('.pth')]
        
        if not model_files:
            raise ValueError("No TabM model file found")
        
        latest_model = sorted(model_files)[-1]
        model_file = os.path.join(load_path, latest_model)
        
        checkpoint = torch.load(model_file, map_location=self.device)
        
        arch = checkpoint['model_architecture']
        self.model = TabMNetwork(
            n_features=arch['n_features'],
            embed_dim=arch['embed_dim'],
            dropout=arch['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])