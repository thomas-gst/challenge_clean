import torch.nn as nn 
import torch.nn.functional as F
import torch

class flashy_Huber(nn.Module):
    '''
    Loss function wrapper for training the flashy-encoder. 
    Huber was used to prevent the model from over-reacting to outliers
    '''
    def __init__(self):
        super(flashy_Huber, self).__init__()
        self.hub_loss = nn.HuberLoss(reduction='mean')

    def forward(self, output, target):
        return self.hub_loss(output, target['log_flashiness']) 

class regressor_MSE(nn.Module):
    '''
    Loss function wrapper for training regressors
    '''
    def __init__(self):
        super(regressor_MSE, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        return self.mse_loss(output, target['log_views']) 
    
    
class multi_task_loss(nn.Module):
    '''
    Custom loss to train multi-task (classification & regression) models. 
    The alpha parameter decides the mix in loss contribution between classification and regresssion,
    in practice it's usually set to 0 or 1 as training regression and classification simultaneously led to inconclusive results.
    The anomaly flag is used to choose what kind of class we're predicting (anomaly class or views class)
    '''
    def __init__(self, alpha=0, num_classes=12, min_anomaly_class=-5, task_perdiod=5, anomaly=True):
        super(multi_task_loss, self).__init__()
        self.mse_loss = nn.HuberLoss(reduction='mean')#nn.MSELoss(reduction='mean')#
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.alpha = alpha
        self.task_perdiod = task_perdiod
        self.num_classes = num_classes
        self.min_anomaly_class = min_anomaly_class
        self.anomaly = anomaly
        

    def forward(self, output, target):
        predicted_log_views, classifier_output = output
        actual_log_views = target['log_views']
        if self.anomaly:
            actual_class = target['encoded_anomaly_class']
        else:
            actual_class = target['encoded_views_class']
        regression_loss = self.mse_loss(predicted_log_views, actual_log_views)
        class_loss = self.ce_loss(classifier_output, actual_class)
        return self.alpha * regression_loss + (1 - self.alpha) * class_loss
    
    def update_alpha(self, epoch, avg_train_loss, avg_val_loss):
        '''
        Update the alpha value for the loss function.
        '''
        if (epoch+1) in self.task_perdiod:
            self.alpha = 0
        else:
            self.alpha = 1
            
    def update_alpha_periodic(self, epoch):
        if (epoch+1)%self.task_perdiod == 0:
            self.alpha = 1-self.alpha
    
class multi_task_distance_loss(nn.Module):
    '''
    Custom distance-based classification loss :
    Since classes with similar labels (ie the distance is small) are similar, the idea
    was to penalize accordingly, ie penalize the missclassification of class 1 as 4 more
    than the missclassification of class 1 as 2. Results were inconclusive 
    '''
    def __init__(self, alpha=0, num_classes=12, min_anomaly_class=-5, task_perdiod=5):
        super(multi_task_distance_loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.alpha = alpha
        self.task_perdiod = task_perdiod

        self.num_classes = num_classes 
        self.min_anomaly_class = min_anomaly_class       
        self.cost_matrix = self._build_cost_matrix(num_classes)
    
    def _build_cost_matrix(self, num_classes):
        '''
        Helper function to build a custom cost matrix for the anomaly classes.
        The cost matrix allows the loss to penalize misclassifications based on the distance between classes.
        '''
        i = torch.arange(num_classes).unsqueeze(1) 
        j = torch.arange(num_classes).unsqueeze(0) 
        return (i - j).pow(2).float() 
    
    def _get_anomaly_class_idx(self, anomaly_class):
        '''
        Helper function to convert the anomaly class tensor to an index suitable for the cost matrix.
        '''
        return anomaly_class - self.min_anomaly_class        
    
    def forward(self, output, target):
        predicted_log_views, classifier_output = output
        actual_log_views, anomaly_class = target['log_views'], target['anomaly_class']
        probabilites = F.softmax(classifier_output, dim=1)
        true_class_idx = self._get_anomaly_class_idx(anomaly_class)
        sample_costs = self.cost_matrix.to(predicted_log_views.device)[true_class_idx, :]
        penalized_probabilities = sample_costs * -torch.log(probabilites + 1e-8)
        class_loss = penalized_probabilities.sum(dim=1).mean() 
        regression_loss = self.mse_loss(predicted_log_views, actual_log_views)
        return self.alpha * regression_loss + (1 - self.alpha) * class_loss
        
    def update_alpha(self, epoch, avg_train_loss, avg_val_loss):
        '''
        Update the alpha value for the loss function.
        '''
        if (epoch+1) % self.task_perdiod == 0:
            self.alpha = 1 - self.alpha
        
        
    
    