import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class UnlearningEvaluator:
    """Graph neural network unlearning effectiveness evaluator"""
    
    def __init__(self, device):
        self.device = device
    
    def evaluate_model_performance(self, model, data, train_mask, val_mask, test_mask):
        """Evaluate basic model performance metrics"""
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.max(1)[1]
            
            train_acc = self.accuracy(pred[train_mask], data.y[train_mask])
            val_acc = self.accuracy(pred[val_mask], data.y[val_mask]) if val_mask.sum() > 0 else 0.0
            test_acc = self.accuracy(pred[test_mask], data.y[test_mask])
            
            train_f1_macro = self.f1_score(pred[train_mask], data.y[train_mask], average='macro')
            train_f1_micro = self.f1_score(pred[train_mask], data.y[train_mask], average='micro')
            
            test_f1_macro = self.f1_score(pred[test_mask], data.y[test_mask], average='macro')
            test_f1_micro = self.f1_score(pred[test_mask], data.y[test_mask], average='micro')
        
        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_f1_macro': train_f1_macro,
            'train_f1_micro': train_f1_micro,
            'test_f1_macro': test_f1_macro,
            'test_f1_micro': test_f1_micro
        }
    
    def evaluate_unlearning_effectiveness(self, original_model, unlearned_model, data, 
                                        unlearn_nodes, remaining_nodes=None):
        """Evaluate unlearning effectiveness"""
        original_model.eval()
        unlearned_model.eval()
        
        with torch.no_grad():
            original_out = F.softmax(original_model(data.x, data.edge_index), dim=1)
            unlearned_out = F.softmax(unlearned_model(data.x, data.edge_index), dim=1)
            
            unlearn_original = original_out[unlearn_nodes]
            unlearn_new = unlearned_out[unlearn_nodes]
            
            kl_div = self.kl_divergence(unlearn_new, unlearn_original)
            
            entropy = self.prediction_entropy(unlearn_new)
            max_entropy = np.log(unlearn_new.shape[1])
            normalized_entropy = entropy / max_entropy
            
            original_confidence = torch.max(unlearn_original, dim=1)[0].mean().item()
            new_confidence = torch.max(unlearn_new, dim=1)[0].mean().item()
            confidence_drop = original_confidence - new_confidence
            
            original_pred = original_out[unlearn_nodes].max(1)[1]
            new_pred = unlearned_out[unlearn_nodes].max(1)[1]
            prediction_change_rate = (original_pred != new_pred).float().mean().item()
            
            unlearning_metrics = {
                'kl_divergence': kl_div,
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'confidence_drop': confidence_drop,
                'original_confidence': original_confidence,
                'new_confidence': new_confidence,
                'prediction_change_rate': prediction_change_rate
            }
            
            if remaining_nodes is not None:
                remaining_original = original_out[remaining_nodes]
                remaining_new = unlearned_out[remaining_nodes]
                
                remaining_kl_div = self.kl_divergence(remaining_new, remaining_original)
                remaining_prediction_change = (
                    remaining_original.max(1)[1] != remaining_new.max(1)[1]
                ).float().mean().item()
                
                unlearning_metrics.update({
                    'remaining_kl_divergence': remaining_kl_div,
                    'remaining_prediction_change_rate': remaining_prediction_change
                })
        
        return unlearning_metrics
    
    def membership_inference_attack(self, target_model, shadow_model, data, 
                                  unlearn_nodes, remaining_nodes):
        """Membership inference attack - evaluate privacy protection effectiveness"""
        target_model.eval()
        shadow_model.eval()
        
        with torch.no_grad():
            target_out = F.softmax(target_model(data.x, data.edge_index), dim=1)
            shadow_out = F.softmax(shadow_model(data.x, data.edge_index), dim=1)
            
            def extract_mia_features(pred_probs, true_labels):
                features = []
                for i, label in enumerate(true_labels):
                    prob_features = pred_probs[i].cpu().numpy()
                    
                    max_prob = torch.max(pred_probs[i]).item()
                    entropy = -torch.sum(pred_probs[i] * torch.log(pred_probs[i] + 1e-8)).item()
                    label_prob = pred_probs[i][label].item() if label < pred_probs[i].size(0) else 0.0
                    
                    feature_vector = np.concatenate([
                        prob_features, 
                        [max_prob, entropy, label_prob]
                    ])
                    features.append(feature_vector)
                
                return np.array(features)
            
            unlearn_features = extract_mia_features(
                target_out[unlearn_nodes], 
                data.y[unlearn_nodes]
            )
            unlearn_labels = np.zeros(len(unlearn_nodes))
            
            if remaining_nodes is not None:
                remaining_features = extract_mia_features(
                    target_out[remaining_nodes],
                    data.y[remaining_nodes] 
                )
                remaining_labels = np.ones(len(remaining_nodes))
                
                all_features = np.concatenate([unlearn_features, remaining_features])
                all_labels = np.concatenate([unlearn_labels, remaining_labels])
            else:
                all_features = unlearn_features
                all_labels = unlearn_labels
            
            if len(np.unique(all_labels)) < 2:
                return {'mia_auc': 0.5, 'mia_acc': 0.5}
            
            X_train, X_test, y_train, y_test = train_test_split(
                all_features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
            )
            
            mia_classifier = LogisticRegression(random_state=42, max_iter=1000)
            mia_classifier.fit(X_train, y_train)
            
            y_pred_proba = mia_classifier.predict_proba(X_test)[:, 1]
            y_pred = mia_classifier.predict(X_test)
            
            mia_auc = roc_auc_score(y_test, y_pred_proba)
            mia_acc = accuracy_score(y_test, y_pred)
        
            return {
                'mia_auc': mia_auc,
                'mia_acc': mia_acc
            }
    
    def utility_preservation_evaluation(self, original_model, unlearned_model, data, test_mask):
        """Utility preservation evaluation - measure model performance retention on test set after unlearning"""
        original_performance = self.evaluate_model_performance(
            original_model, data, test_mask, test_mask, test_mask
        )
        
        unlearned_performance = self.evaluate_model_performance(
            unlearned_model, data, test_mask, test_mask, test_mask
        )
        
        utility_metrics = {}
        for metric in ['test_acc', 'test_f1_macro', 'test_f1_micro']:
            original_value = original_performance[metric]
            unlearned_value = unlearned_performance[metric]
            utility_drop = original_value - unlearned_value
            utility_retention = unlearned_value / original_value if original_value > 0 else 1.0
            
            utility_metrics[f'{metric}_original'] = original_value
            utility_metrics[f'{metric}_unlearned'] = unlearned_value
            utility_metrics[f'{metric}_drop'] = utility_drop
            utility_metrics[f'{metric}_retention'] = utility_retention
        
        return utility_metrics
    
    def comprehensive_evaluation(self, original_model, unlearned_model, data, 
                               train_mask, val_mask, test_mask, unlearn_nodes):
        """Comprehensive evaluation"""
        remaining_nodes = torch.where(train_mask)[0]
        remaining_nodes = remaining_nodes[~torch.isin(remaining_nodes, unlearn_nodes)]
        
        performance_metrics = self.evaluate_model_performance(
            unlearned_model, data, train_mask, val_mask, test_mask
        )
        
        unlearning_metrics = self.evaluate_unlearning_effectiveness(
            original_model, unlearned_model, data, unlearn_nodes, remaining_nodes
        )
        
        utility_metrics = self.utility_preservation_evaluation(
            original_model, unlearned_model, data, test_mask
        )
        
        mia_metrics = self.membership_inference_attack(
            unlearned_model, original_model, data, unlearn_nodes, remaining_nodes
        )
        
        all_metrics = {
            **performance_metrics,
            **unlearning_metrics, 
            **utility_metrics,
            **mia_metrics
        }
        
        return all_metrics
    
    def accuracy(self, pred, target):
        """Calculate accuracy"""
        return (pred == target).float().mean().item()
    
    def f1_score(self, pred, target, average='macro'):
        """Calculate F1 score"""
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        return f1_score(target_np, pred_np, average=average, zero_division=0)
    
    def kl_divergence(self, p, q):
        """Calculate KL divergence"""
        return F.kl_div(
            torch.log(p + 1e-8), 
            q, 
            reduction='batchmean'
        ).item()
    
    def prediction_entropy(self, probs):
        """Calculate prediction entropy"""
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()


def evaluate_unlearning_method(unlearning_method, data_processor, original_model, 
                              unlearn_nodes, unlearn_edges, device):
    """Evaluate the effectiveness of a specific unlearning method"""
    evaluator = UnlearningEvaluator(device)
    
    if unlearn_nodes is not None:
        result = unlearning_method.node_unlearn(data_processor, original_model, unlearn_nodes)
    elif unlearn_edges is not None:
        result = unlearning_method.edge_unlearn(data_processor, original_model, unlearn_edges)
    else:
        raise ValueError("Either unlearn_nodes or unlearn_edges must be provided")
    
    if isinstance(result[0], list):
        shard_models, unlearn_time = result
        from unlearning_func import GraphEraserWrapper
        unlearned_model = GraphEraserWrapper(shard_models, unlearning_method.aggregation)
    else:
        unlearned_model, unlearn_time = result
    
    metrics = evaluator.comprehensive_evaluation(
        original_model, unlearned_model, 
        data_processor.data.to(device),
        data_processor.train_mask,
        data_processor.val_mask, 
        data_processor.test_mask,
        unlearn_nodes if unlearn_nodes is not None else torch.tensor([]).to(device)
    )
    
    metrics['unlearning_time'] = unlearn_time
    
    return metrics, unlearned_model