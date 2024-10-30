---
layout: post
title:  "Federated Learning in Healthcare: A Practical Guide

"
date:   2024-10-28 
categories: AI
---
# Federated Learning in Healthcare: A Practical Guide

## Why Federated Learning for Healthcare?

Healthcare data is:
1. Highly sensitive (HIPAA, GDPR)
2. Siloed across hospitals
3. Valuable for research and improving patient care

Federated learning allows hospitals to collaborate without sharing raw patient data.

## Practical Implementation

### 1. Basic Medical Image Classification

Let's start with a simplified example of classifying X-ray images across multiple hospitals:

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class Hospital:
    def __init__(self, hospital_id, x_ray_data, labels):
        self.id = hospital_id
        self.data = x_ray_data  # Simplified as numerical features
        self.labels = labels    # Binary: 0 for normal, 1 for abnormal
        self.scaler = StandardScaler()
        
    def preprocess_data(self):
        return self.scaler.fit_transform(self.data)
    
    def train_local_model(self, global_model=None):
        X = self.preprocess_data()
        
        if global_model is None:
            local_model = MLPClassifier(hidden_layer_sizes=(100, 50),
                                       max_iter=10)
        else:
            local_model = clone_model(global_model)
        
        local_model.fit(X, self.labels)
        return get_model_params(local_model)
    
    def evaluate_model(self, model):
        X = self.preprocess_data()
        return model.score(X, self.labels)

def clone_model(model):
    new_model = MLPClassifier(hidden_layer_sizes=model.hidden_layer_sizes)
    new_model.coefs_ = [w.copy() for w in model.coefs_]
    new_model.intercepts_ = [b.copy() for b in model.intercepts_]
    return new_model

def get_model_params(model):
    return {
        "coefs": [w.copy() for w in model.coefs_],
        "intercepts": [b.copy() for b in model.intercepts_]
    }

def create_global_model(params_list):
    # Average the parameters from all hospitals
    avg_coefs = [
        np.mean([p["coefs"][i] for p in params_list], axis=0)
        for i in range(len(params_list[0]["coefs"]))
    ]
    avg_intercepts = [
        np.mean([p["intercepts"][i] for p in params_list], axis=0)
        for i in range(len(params_list[0]["intercepts"]))
    ]
    
    global_model = MLPClassifier(hidden_layer_sizes=(100, 50))
    global_model.coefs_ = avg_coefs
    global_model.intercepts_ = avg_intercepts
    return global_model

# Simulating multiple hospitals
def simulate_federated_learning():
    # Create simulated hospital data
    hospitals = [
        Hospital("H1", np.random.rand(1000, 200), np.random.randint(2, size=1000)),
        Hospital("H2", np.random.rand(800, 200), np.random.randint(2, size=800)),
        Hospital("H3", np.random.rand(1200, 200), np.random.randint(2, size=1200))
    ]
    
    global_model = None
    
    for round in range(5):
        print(f"Training Round {round + 1}")
        
        # Local training at each hospital
        local_models = []
        for hospital in hospitals:
            local_params = hospital.train_local_model(global_model)
            local_models.append(local_params)
        
        # Aggregate models
        global_model = create_global_model(local_models)
        
        # Evaluate global model at each hospital
        for hospital in hospitals:
            accuracy = hospital.evaluate_model(global_model)
            print(f"Hospital {hospital.id} accuracy: {accuracy:.4f}")
    
    return global_model

# Run the simulation
final_model = simulate_federated_learning()
```

### 2. Advanced Implementation: Patient Outcome Prediction

Now let's look at a more complex example predicting patient outcomes:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np

class MedicalCenter:
    def __init__(self, center_id, patient_data):
        self.id = center_id
        self.data = patient_data
        self.feature_columns = [col for col in patient_data.columns 
                               if col not in ['patient_id', 'outcome']]
    
    def preprocess_data(self):
        # Handle missing values
        for col in self.feature_columns:
            if self.data[col].dtype in ['int64', 'float64']:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            else:
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        for col in self.feature_columns:
            if self.data[col].dtype == 'object':
                self.data[col] = pd.Categorical(self.data[col]).codes
        
        return self.data[self.feature_columns], self.data['outcome']
    
    def train_local_model(self, global_model=None):
        X, y = self.preprocess_data()
        
        if global_model is None:
            local_model = RandomForestClassifier(n_estimators=100)
        else:
            local_model = global_model
        
        local_model.fit(X, y)
        return local_model
    
    def evaluate_model(self, model):
        X, y = self.preprocess_data()
        y_pred_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        
        return {
            'auc': auc,
            'precision': precision,
            'recall': recall
        }

def federated_model_averaging(models):
    # Average the predictions from all models
    def averaged_predict_proba(X):
        predictions = np.zeros((X.shape[0], 2))
        for model in models:
            predictions += model.predict_proba(X)
        return predictions / len(models)
    
    # Create a new model with the averaged predict_proba method
    averaged_model = RandomForestClassifier()
    averaged_model.classes_ = models[0].classes_
    averaged_model.predict_proba = averaged_predict_proba
    
    return averaged_model

# Simulate federated learning with medical centers
def run_medical_federated_learning():
    # Create simulated medical center data
    def generate_patient_data(num_patients):
        return pd.DataFrame({
            'patient_id': range(num_patients),
            'age': np.random.randint(18, 90, num_patients),
            'blood_pressure': np.random.randint(90, 180, num_patients),
            'glucose': np.random.randint(70, 200, num_patients),
            'heart_rate': np.random.randint(60, 100, num_patients),
            'gender': np.random.choice(['M', 'F'], num_patients),
            'smoker': np.random.choice(['Yes', 'No'], num_patients),
            'outcome': np.random.randint(2, size=num_patients)
        })

    medical_centers = [
        MedicalCenter("MC1", generate_patient_data(1000)),
        MedicalCenter("MC2", generate_patient_data(800)),
        MedicalCenter("MC3", generate_patient_data(1200))
    ]
    
    for round in range(3):
        print(f"\nTraining Round {round + 1}")
        
        local_models = []
        for center in medical_centers:
            local_model = center.train_local_model()
            local_models.append(local_model)
        
        # Create global model
        global_model = federated_model_averaging(local_models)
        
        # Evaluate global model at each center
        for center in medical_centers:
            metrics = center.evaluate_model(global_model)
            print(f"Medical Center {center.id} - AUC: {metrics['auc']:.4f}")
    
    return global_model

# Feature importance analysis
def analyze_feature_importance(model, feature_names):
    importances = np.mean([tree.feature_importances_ 
                           for tree in model.estimators_], axis=0)
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    return feature_imp.sort_values('importance', ascending=False)

# Run the simulation
final_medical_model = run_medical_federated_learning()
```

### 3. Privacy-Preserving Patient Similarity Analysis

This example shows how to find similar patients across hospitals without sharing raw data:

```python
import numpy as np
from scipy.spatial.distance import cosine
import hashlib

class PrivatePatientSimilarity:
    def __init__(self, hospital_id):
        self.id = hospital_id
        self.local_patients = {}
    
    def add_patient(self, patient_id, features):
        # Hash patient ID for privacy
        hashed_id = hashlib.sha256(str(patient_id).encode()).hexdigest()
        self.local_patients[hashed_id] = features
    
    def generate_similarity_matrix(self):
        patient_ids = list(self.local_patients.keys())
        n_patients = len(patient_ids)
        similarity_matrix = np.zeros((n_patients, n_patients))
        
        for i in range(n_patients):
            for j in range(i+1, n_patients):
                sim = 1 - cosine(self.local_patients[patient_ids[i]], 
                                self.local_patients[patient_ids[j]])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return similarity_matrix, patient_ids
    
    def find_similar_patients(self, query_features, top_k=5):
        similarities = []
        for patient_id, features in self.local_patients.items():
            sim = 1 - cosine(query_features, features)
            similarities.append((patient_id, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

class FederatedPatientSimilarity:
    def __init__(self, hospitals):
        self.hospitals = hospitals
    
    def find_similar_patients_federated(self, query_features, top_k=5):
        all_similarities = []
        
        for hospital in self.hospitals:
            local_similarities = hospital.find_similar_patients(query_features, top_k)
            all_similarities.extend([(hospital.id, *sim) for sim in local_similarities])
        
        return sorted(all_similarities, key=lambda x: x[2], reverse=True)[:top_k]

# Example usage
def simulate_patient_similarity():
    # Create hospitals with simulated patient data
    hospitals = [
        PrivatePatientSimilarity(f"Hospital_{i}") 
        for i in range(3)
    ]
    
    # Add simulated patients to each hospital
    for hospital in hospitals:
        for i in range(100):
            patient_features = np.random.rand(50)  # 50 medical features
            hospital.add_patient(f"Patient_{i}", patient_features)
    
    # Create federated system
    federated_system = FederatedPatientSimilarity(hospitals)
    
    # Query similar patients
    query_features = np.random.rand(50)  # New patient features
    similar_patients = federated_system.find_similar_patients_federated(query_features)
    
    return similar_patients

# Run simulation
similar_patients = simulate_patient_similarity()
print("\nMost similar patients across all hospitals:")
for hospital_id, patient_id, similarity in similar_patients:
    print(f"Hospital: {hospital_id}, Patient: {patient_id}, Similarity: {similarity:.4f}")
```

## Practical Considerations for Medical Federated Learning

### 1. Data Standardization

Different hospitals may use different scales or units. Here's how to handle it:

```python
class DataStandardizer:
    def __init__(self):
        self.feature_ranges = {}
    
    def update_ranges(self, hospital_data):
        for feature in hospital_data.columns:
            if feature not in self.feature_ranges:
                self.feature_ranges[feature] = {'min': float('inf'), 'max': float('-inf')}
            
            local_min = hospital_data[feature].min()
            local_max = hospital_data[feature].max()
            
            self.feature_ranges[feature]['min'] = min(self.feature_ranges[feature]['min'], local_min)
            self.feature_ranges[feature]['max'] = max(self.feature_ranges[feature]['max'], local_max)
    
    def get_standardization_params(self):
        return self.feature_ranges

class Hospital:
    def standardize_data(self, standardizer_params):
        standardized_data = self.data.copy()
        for feature, range_info in standardizer_params.items():
            if feature in standardized_data:
                min_val = range_info['min']
                max_val = range_info['max']
                standardized_data[feature] = (standardized_data[feature] - min_val) / (max_val - min_val)
        return standardized_data
```

### 2. Handling Missing Data

Medical data often has missing values. Here's a robust approach:

```python
class MissingDataHandler:
    def __init__(self):
        self.imputation_values = {}
    
    def calculate_imputation_values(self, hospitals):
        all_values = {}
        for hospital in hospitals:
            for column in hospital.data.columns:
                if column not in all_values:
                    all_values[column] = []
                all_values[column].extend(hospital.data[column].dropna().tolist())
        
        for column, values in all_values.items():
            if len(values) > 0:
                if isinstance(values[0], (int, float)):
                    self.imputation_values[column] = np.median(values)
                else:
                    self.imputation_values[column] = max(set(values), key=values.count)
    
    def impute_missing_values(self, data):
        imputed_data = data.copy()
        for column, value in self.imputation_values.items():
            if column in imputed_data:
                imputed_data[column].fillna(value, inplace=True)
        return imputed_data
```

### 3. Model Evaluation Metrics for Medical Data

```python
class MedicalModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred_proba):
        thresholds = np.arange(0, 1.1, 0.1)
        metrics = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            TP = np.sum((y_true == 1) & (y_pred == 1))
            TN = np.sum((y_true == 0) &
