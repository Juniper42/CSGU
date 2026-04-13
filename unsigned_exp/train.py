import time
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np


class Trainer:
    """GNN model trainer"""
    
    def __init__(self, model, data, args, device):
        self.model = model.to(device)
        self.data = data.to(device) 
        self.args = args
        self.device = device
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        self.criterion = F.nll_loss
        
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
    def train_epoch(self, train_mask):
        """Train one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out[train_mask], self.data.y[train_mask])
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, mask):
        """Evaluate model performance"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out[mask].max(1)[1]
            correct = pred.eq(self.data.y[mask]).double()
            accuracy = correct.sum() / len(correct)
        return accuracy.item()
    
    def get_predictions_and_embeddings(self, mask=None):
        """Get predictions and node embeddings"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            embeddings = self.model.get_embeddings(self.data.x, self.data.edge_index)
            
            if mask is not None:
                out = out[mask]
                embeddings = embeddings[mask]
                
            predictions = F.softmax(out, dim=1)
        
        return predictions, embeddings
    
    def train(self, train_mask, val_mask=None, verbose=True):
        """Complete training process"""
        train_losses = []
        val_accuracies = []
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(train_mask)
            train_losses.append(train_loss)
            
            if val_mask is not None and epoch % self.args.eval_step == 0:
                val_acc = self.evaluate(val_mask)
                val_accuracies.append(val_acc)
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if verbose and epoch % (self.args.epochs // 10) == 0:
                    print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                if self.patience_counter >= self.args.patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}')
                    break
            
            elif verbose and epoch % (self.args.epochs // 10) == 0:
                train_acc = self.evaluate(train_mask)
                print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f'Training completed in {training_time:.2f}s')
            if val_mask is not None:
                print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': self.best_val_acc,
            'training_time': training_time
        }
    
    def save_model(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, path)
    
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']


def train_model(model, data, train_mask, val_mask, test_mask, args, device, verbose=True):
    """Convenience function for training models"""
    trainer = Trainer(model, data, args, device)
    
    train_info = trainer.train(train_mask, val_mask, verbose=verbose)
    
    test_acc = trainer.evaluate(test_mask)
    train_acc = trainer.evaluate(train_mask)
    val_acc = trainer.evaluate(val_mask) if val_mask is not None else 0.0
    
    results = {
        'train_acc': train_acc,
        'val_acc': val_acc, 
        'test_acc': test_acc,
        'training_time': train_info['training_time'],
        'trainer': trainer
    }
    
    if verbose:
        print(f'Final Results - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    return results


class MultiRunTrainer:
    """Multi-run trainer for obtaining stable results"""
    
    def __init__(self, args):
        self.args = args
        self.results = []
    
    def run_multiple_experiments(self, model_class, data_processor, device, verbose=True):
        """Run multiple experiments"""
        all_results = []
        
        for run in range(self.args.runs):
            if verbose:
                print(f"\n=== Run {run + 1}/{self.args.runs} ===")
            
            torch.manual_seed(self.args.seed + run)
            np.random.seed(self.args.seed + run)
            
            data = data_processor.data
            train_mask, val_mask, test_mask = data_processor.create_train_test_split()
            
            model = model_class(
                num_features=data.x.shape[1],
                num_classes=data_processor.num_classes[self.args.dataset],
                hidden_dim=self.args.hidden_dim,
                num_layers=self.args.num_layers,
                dropout=self.args.dropout
            )
            
            results = train_model(
                model, data, train_mask, val_mask, test_mask, 
                self.args, device, verbose=verbose
            )
            
            all_results.append(results)
        
        avg_results = self.calculate_average_results(all_results)
        
        if verbose:
            self.print_summary(avg_results)
        
        return avg_results, all_results
    
    def calculate_average_results(self, results):
        """Calculate average results from multiple runs"""
        metrics = ['train_acc', 'val_acc', 'test_acc', 'training_time']
        avg_results = {}
        
        for metric in metrics:
            values = [r[metric] for r in results]
            avg_results[f'{metric}_mean'] = np.mean(values)
            avg_results[f'{metric}_std'] = np.std(values)
        
        return avg_results
    
    def print_summary(self, avg_results):
        """Print results summary"""
        print("\n=== Summary ===")
        print(f"Train Acc: {avg_results['train_acc_mean']:.4f} ± {avg_results['train_acc_std']:.4f}")
        print(f"Val Acc: {avg_results['val_acc_mean']:.4f} ± {avg_results['val_acc_std']:.4f}")  
        print(f"Test Acc: {avg_results['test_acc_mean']:.4f} ± {avg_results['test_acc_std']:.4f}")
        print(f"Training Time: {avg_results['training_time_mean']:.2f}s ± {avg_results['training_time_std']:.2f}s")