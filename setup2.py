import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, install if not available
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    print("XGBoost not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    from xgboost import XGBClassifier
    xgboost_available = True

# Load and prepare the dataset (same as your original code)
def prepare_dataset():
    """Prepare the dataset for model training."""
    np.random.seed(42)
    n_samples = 1000

    class_dist = {
        'e-waste': 0.118,
        'cardboard': 0.106,
        'glass': 0.113,
        'metal': 0.107,
        'organic': 0.115,
        'paper': 0.101,
        'plastic': 0.106,
        'textile': 0.117,
        'trash': 0.115,
    }
    # Generate class labels
    classes = list(class_dist.keys())
    probabilities = list(class_dist.values())
    # Normalize probabilities to sum to 1
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    labels = np.random.choice(classes, size=n_samples, p=probabilities)
     
    data = []

    for label in labels:
        if label == 'e-waste':
            aspect_ratio = np.random.uniform(0.9, 2.0)
        elif label == 'cardboard':
            aspect_ratio = np.random.uniform(0.9, 2.0)
        elif label == 'glass':
            aspect_ratio = np.random.uniform(0.6, 2.5)
        elif label == 'metal':
            aspect_ratio = np.random.uniform(0.9, 2.7)
        elif label == 'organic':
            aspect_ratio = np.random.uniform(0.8, 3.2)
        elif label == 'paper':
            aspect_ratio = np.random.uniform(0.7, 3.1)
        elif label == 'plastic':
            aspect_ratio = np.random.uniform(0.8, 2.1)
        elif label == 'textile':
            aspect_ratio = np.random.uniform(0.7, 1.0)
        elif label == 'trash':
            aspect_ratio = np.random.uniform(0.9, 2.2)
        
        if label == 'e-waste' or label == 'trash':
            blur = np.random.uniform(200, 18000)
        else:
            blur = np.random.uniform(200, 15000)
            
        brightness_map = {
            'e-waste': (100, 290),
            'trash': (140, 265),
            'cardboard': (50, 175),
            'glass': (5, 25),
            'metal': (160, 260),
            'organic': (150, 245),
            'paper': (105, 190),
            'plastic': (100, 185),
            'textile': (145, 255),
        }
        
        key = label if label in brightness_map else 'trash' if label == 'e-waste' else label
        b_min, b_max = brightness_map.get(key, (100, 200))
        brightness = np.random.uniform(b_min, b_max)
        
        contrast_map = {
            'e-waste': (30, 80),
            'cardboard': (40, 70),
            'glass': (30, 60),
            'metal': (35, 65),
            'organic': (30, 60),
            'paper': (25, 55),
            'plastic': (20, 50),
            'textile': (15, 45),
            'trash': (10, 40),
        }
        
        c_min, c_max = contrast_map.get(label, (20, 60))
        contrast = np.random.uniform(c_min, c_max)
        
        edge_map = {
            'e-waste': 0.045,
            'cardboard': 0.050,
            'glass': 0.040, 
            'metal': 0.045,
            'organic': 0.040,
            'paper': 0.045,
            'plastic': 0.040,
            'textile': 0.045,
            'trash': 0.040
        }
        base_edge = edge_map.get(label, 0.042)
        edge_density = np.random.uniform(base_edge, 0.005)
        edge_density = max(0.02, min(0.08, edge_density))
        
        entropy_map = {
            'e-waste': (6.5, 7.5),
            'cardboard': (4.7, 7.5),
            'glass': (2.8, 7.9),  
            'metal': (5.45, 7.9),
            'organic': (6.15, 7.5),
            'paper': (5.65, 7.5),
            'plastic': (6.5, 7.5), 
            'textile': (5.65, 7.5),
            'trash': (4.0, 7.5) 
        }
        e_min, e_max = entropy_map.get(label, (5.0, 7.5))
        entropy = np.random.uniform(e_min, e_max)
        
        width = np.random.choice([100, 200, 300, 400, 500, 600, 640, 800])
        height = np.random.choice([100, 150, 200, 250, 300, 350, 400, 480, 600])
        
        file_size = (width * height) / 1000 * np.random.uniform(0.8, 1.2)
        
        data.append({
            'label': label,
            'aspect_ratio': aspect_ratio,
            'blur': blur,
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'entropy': entropy,
            'width': width,
            'height': height,
            'file_size': file_size
        })

    df = pd.DataFrame(data)
    return df   

print("Generating dataset from statistical information...")
df = prepare_dataset()
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")    
print(df['label'].value_counts(normalize=True)) 

def engineer_features(df):
    df_feat = df.copy()
    df_feat['aspect_ratio_squared'] = df_feat['aspect_ratio'] ** 2
    
    df_feat['resolution'] = df_feat['width'] * df_feat['height']
    df_feat['resolution_category'] = pd.cut(df_feat['resolution'], bins=[0, 50000, 150000, 300000, 1000000], labels=['low', 'medium', 'high', 'very_high'])
       
    # Interaction features 
    df_feat['brightness_contrast_ratio'] = df_feat['brightness'] / (df_feat['contrast'] + 1)
    df_feat['edge_entropy_interaction'] = df_feat['edge_density'] * df_feat['entropy']
    df_feat['blur_edge_interaction'] = df_feat['blur'] * df_feat['edge_density']
    df_feat['texture_complexity'] = df_feat['entropy'] / (df_feat['edge_density'] * 100) / (df_feat['blur'] / 100 + 1)
    
    res_dummies = pd.get_dummies(df_feat['resolution_category'], prefix='res')
    df_feat = pd.concat([df_feat, res_dummies], axis=1)
    return df_feat

df_feat = engineer_features(df)
print(f"\nFeature after engineering: {df_feat.shape[1]}")   

feature_columns = ['aspect_ratio', 'aspect_ratio_squared', 'blur', 'brightness', 'contrast', 'edge_density', 'entropy', 'width', 'height', 'file_size', 'brightness_contrast_ratio', 'edge_entropy_interaction', 'blur_edge_interaction', 'texture_complexity', 'res_low', 'res_medium', 'res_high', 'res_very_high']   
   
x = df_feat[feature_columns]
y = df_feat['label']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x) 
x_scaled = pd.DataFrame(x_scaled, columns=feature_columns)
                
y_binary = (y == 'e-waste').astype(int)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary)           
  
print(f"Training set: {x_train.shape}")
print(f"Test set: {x_test.shape}")
print(f"E-waste proportion in training: {y_train.mean():.3f}")

# MODEL 1: XGBoost Classifier (instead of Gradient Boosting)
# MODEL 2: Neural Network (MLPClassifier) (instead of SVM)
binary_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                                     max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
}
best_binary_model = None
best_binary_score = 0   
binary_results = {}

for name, model in binary_models.items():
    print(f"\n{'='*50}")
    print(f"Training {name} for Binary Classification...")
    print(f"{'='*50}")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{name} Accuracy: {accuracy:.4f}")   
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['non-e-waste', 'e-waste']))
    
    cv_scores = cross_val_score(model, x_scaled, y_binary, cv=5)
    print(f"Cross-validation scores: {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")
    
    binary_results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    if accuracy > best_binary_score:
        best_binary_score = accuracy
        best_binary_model = model
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            plt.figure(figsize=(12, 6))
            indices = np.argsort(importances)[::-1] 
            
            plt.title(f'Feature Importance for Binary Classification ({name})')
            plt.bar(range(len(importances[:15])), importances[indices[:15]])
            plt.xticks(range(len(importances[:15])), [feature_columns[i] for i in indices[:15]], rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

print("\n" + "="*60)
print("Model 2: Multi-class Classifier for all 9 waste types")
print("="*60)

label_encoder = LabelEncoder()
y_multi = label_encoder.fit_transform(y)

x_train_m, x_test_m, y_train_multi, y_test_m = train_test_split(x_scaled, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
print(f"Classes: {label_encoder.classes_}")
print(f"Training set: {x_train_m.shape}")
print(f"Test set: {x_test_m.shape}")  

# Multi-class models - using XGBoost and Neural Network
multi_models = {
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=8, 
                             random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(150, 100, 50), activation='relu', 
                                    solver='adam', max_iter=800, random_state=42, 
                                    early_stopping=True, validation_fraction=0.1)
}  

best_multi_model = None
best_multi_score = 0
multi_results = {}

for name, model in multi_models.items():
    print(f"\n{'='*50}")
    print(f"Training {name} for Multi-class Classification...")
    print(f"{'='*50}")
    model.fit(x_train_m, y_train_multi)
    y_pred_multi = model.predict(x_test_m)
    accuracy_multi = accuracy_score(y_test_m, y_pred_multi)
    
    print(f"{name} Accuracy: {accuracy_multi:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_m, y_pred_multi, target_names=label_encoder.classes_))
    
    cv_scores_multi = cross_val_score(model, x_scaled, y_multi, cv=5)
    print(f"Cross-validation scores: {cv_scores_multi.mean():.4f} ± {cv_scores_multi.std() * 2:.4f}")
    
    multi_results[name] = {
        'accuracy': accuracy_multi,
        'cv_mean': cv_scores_multi.mean(),
        'cv_std': cv_scores_multi.std()
    }
    
    if accuracy_multi > best_multi_score:
        best_multi_score = accuracy_multi
        best_multi_model = model
        
        # Confusion Matrix
        plt.figure(figsize=(14, 10))
        y_pred_best = best_multi_model.predict(x_test_m)
        cm = confusion_matrix(y_test_m, y_pred_best)    
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix for Multi-class Classification ({name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()  
        plt.show()
        
        # Feature importance for tree-based models
        if hasattr(best_multi_model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importances = best_multi_model.feature_importances_
            indices = np.argsort(importances)[::-1] 
            
            plt.title(f'Feature Importance for Multi-class Classification ({name})')
            plt.bar(range(len(importances[:15])), importances[indices[:15]])
            plt.xticks(range(len(importances[:15])), [feature_columns[i] for i in indices[:15]], rotation=45, ha='right')
            plt.tight_layout()
            plt.show()  

# Model Comparison
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

print("\nBinary Classification Models:")
binary_compare_df = pd.DataFrame(binary_results).T
print(binary_compare_df.round(4))

print("\nMulti-class Classification Models:")
multi_compare_df = pd.DataFrame(multi_results).T
print(multi_compare_df.round(4))

# Plot model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binary comparison
axes[0].bar(binary_results.keys(), [r['accuracy'] for r in binary_results.values()], alpha=0.7)
axes[0].set_title('Binary Classification Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0.8, 1.0])
axes[0].tick_params(axis='x', rotation=45)

# Multi-class comparison
axes[1].bar(multi_results.keys(), [r['accuracy'] for r in multi_results.values()], alpha=0.7, color='green')
axes[1].set_title('Multi-class Classification Accuracy Comparison')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0.5, 0.9])
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Per-class Performance for best multi-class model
from sklearn.metrics import precision_recall_fscore_support 

y_pred_multi = best_multi_model.predict(x_test_m)
precision, recall, f1, support = precision_recall_fscore_support(y_test_m, y_pred_multi)

print("\n" + "="*60)
print(f"Per-class Performance (Best Model: {type(best_multi_model).__name__})")
print("="*60)
perf_df = pd.DataFrame({
    'Class': label_encoder.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support  
})

print(perf_df.sort_values(by='F1-Score', ascending=False).round(4))

# Feature importance analysis for best binary model
if hasattr(best_binary_model, 'feature_importances_'):
    binary_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_binary_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nTop 10 Features for Binary Classification:")
    print(binary_importance.head(10))

# Prediction function (same as your original)
def predict_waste_type(features, model_type='binary'):
    """
    Predict waste type based on image features
    parameters:
    features: dict with keys like 'aspect_ratio', 'blur', 'brightness', 'contrast', 'edge_density', 'entropy', 'width', 'height', 'file_size'    
    model_type: 'binary' for e-waste detection, 'multi' for multi-class classification
    """
    input_df = pd.DataFrame([features])
    
    input_df['aspect_ratio_squared'] = input_df['aspect_ratio'] ** 2   
    input_df['resolution'] = input_df['width'] * input_df['height']
    input_df['brightness_contrast_ratio'] = input_df['brightness'] / (input_df['contrast'] + 1)
    input_df['edge_entropy_interaction'] = input_df['edge_density'] * input_df['entropy']
    input_df['blur_edge_interaction'] = input_df['blur'] * input_df['edge_density']
    input_df['texture_complexity'] = input_df['entropy'] * (input_df['edge_density'] * 100) / (input_df['blur'] / 100 + 1)
    
    res = input_df['resolution'].values[0]
    input_df['res_low'] = 1 if res <= 50000 else 0
    input_df['res_medium'] = 1 if 50000 < res <= 150000 else 0
    input_df['res_high'] = 1 if 150000 < res <= 300000 else 0
    input_df['res_very_high'] = 1 if res > 300000 else 0
    
    input_features = input_df[feature_columns]
    
    input_scaled = scaler.transform(input_features)
    
    if model_type == 'binary':
        prediction = best_binary_model.predict(input_scaled)[0]
        probability = best_binary_model.predict_proba(input_scaled)[0]
        return {
            'is_e_waste': bool(prediction),
            'probability': probability[1] if prediction == 1 else probability[0],
            'model_used': type(best_binary_model).__name__
        }
    else:
        prediction = best_multi_model.predict(input_scaled)[0]
        probability = best_multi_model.predict_proba(input_scaled)[0]
        class_idx = np.argmax(probability)
        return {
            'predicted_class': label_encoder.inverse_transform([prediction])[0],
            'confidence': probability[class_idx],
            'all_probabilities': dict(zip(label_encoder.classes_, probability)),
            'model_used': type(best_multi_model).__name__
        }

# Example Predictions
example_features = {
    'aspect_ratio': 1.5,
    'blur': 2000,
    'brightness': 250,
    'contrast': 55,
    'edge_density': 0.045,
    'entropy': 7.2,
    'width': 640,
    'height': 480,
    'file_size': 300
}

print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)
print("\nExample 1 (E-waste candidate):")
print(example_features)
binary_pred = predict_waste_type(example_features, 'binary')
multi_pred = predict_waste_type(example_features, 'multi')
print(f"\nBinary Prediction ({binary_pred['model_used']}): {binary_pred}")
print(f"\nMulti-class Prediction ({multi_pred['model_used']}): {multi_pred}")

example_features_2 = {
    'aspect_ratio': 1.3,
    'blur': 800,
    'brightness': 220,
    'contrast': 35,
    'edge_density': 0.040,
    'entropy': 6.8,
    'width': 400,
    'height': 300,
    'file_size': 120
}

print("\nExample 2 (Non E-waste candidate):")
print(example_features_2)
binary_pred2 = predict_waste_type(example_features_2, 'binary')
multi_pred2 = predict_waste_type(example_features_2, 'multi')
print(f"\nBinary Prediction ({binary_pred2['model_used']}): {binary_pred2}")
print(f"\nMulti-class Prediction ({multi_pred2['model_used']}): {multi_pred2}")

print("\n" + "="*60)
print("Training completed! Two NEW models are ready")
print("="*60)
print("1. XGBoost Classifier (replaces Gradient Boosting)")
print("2. Neural Network (MLPClassifier) (replaces SVM)")
print("\nKey Differences from Original Models:")
print("- XGBoost: More advanced gradient boosting with regularization")
print("- Neural Network: Can capture complex non-linear patterns")
print("\nBoth models have been evaluated and compared with Random Forest baseline")