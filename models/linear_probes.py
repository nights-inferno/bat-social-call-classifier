
#from turtle import pd
import pandas as pd

import numpy as np
import random

from sklearn.metrics import average_precision_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from models.classifier_models import BalancedMLP,MultilabelPrevalenceBaseline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.dataset import PipistrelleDataset,AugmentationPipeline
from models.feature_generation import build_feature_bank, extract_encoder,pool_features

def linear_probe_oversample(csv_data, n_split=5, random_state=42, balance=False, encoder_name='perch2'):
    df = pd.read_csv(csv_data)
    label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
    
    kf = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
    clf_names = ['SVM', 'Random Forest', 'MLP']
    y_true_all = []
    y_proba_all = {name: [] for name in clf_names}
    
    encoder = extract_encoder(encoder_name, device='cpu')
    aug_pipeline = AugmentationPipeline()  # For oversampling logic only

    for fold, (train_idx, test_idx) in enumerate(kf.split(df, df[label_cols].values)):
        print(f"Processing Fold {fold+1}...")

        # ── 1. Split FIRST, no oversampling yet ──────────────────────────────
        # is_training=False + resample=False ensures __len__ = unique files only
        train_ds = PipistrelleDataset(df.iloc[train_idx], is_training=False, resample=False, encoder=encoder_name)
        test_ds  = PipistrelleDataset(df.iloc[test_idx],  is_training=False, resample=False, encoder=encoder_name)

        # ── 2. Encode ONCE per unique file ───────────────────────────────────
        # build_feature_bank returns one embedding per window, averaged per file
        # Shape: [n_unique_train_files, embed_dim]
        feature_list_tr, y_train_raw = build_feature_bank(train_ds, encoder, encoder_name, device='cpu')
        X_train_raw = pool_features(
            pool_features(feature_list_tr, windows=True, method='mean',encoder=encoder_name),
              windows=False, window_pooled=True, method='mean',encoder=encoder_name)

        feature_list_ts,y_test = build_feature_bank(test_ds,  encoder, encoder_name, device='cpu')
        X_test = pool_features(
            pool_features(feature_list_ts, windows=True, method='mean',encoder=encoder_name),
              windows=False, window_pooled=True, method='mean',encoder=encoder_name)


        # ── 3. Oversample IN FEATURE SPACE (just numpy row duplication) ──────
        # Zero encoding cost — oversampling is now O(n) memcpy
        X_train, y_train = aug_pipeline.iterative_oversample(X_train_raw, y_train_raw)

        # ── 4. Scale (fit only on pre-oversample or post — both fine,
        #       but pre-oversample is slightly cleaner) ─────────────────────
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        y_true_all.append(y_test)

        models = {
            'SVM': OneVsRestClassifier(SVC(probability=True, random_state=random_state,
                       class_weight='balanced' if balance else None)),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state,
                       class_weight='balanced' if balance else None),
            'MLP': BalancedMLP(input_dim=X_train.shape[1], hidden_dim=128, lr=0.001,
                       epochs=50, dropout=0.2, balanced=balance, batch_norm=True)
        }

        for name, clf in models.items():
            if name == 'MLP':
                # MLP gets the oversampled feature tensor directly
                # Wrap in a TensorDataset so you keep your DataLoader pattern
                mlp_ds = TensorDataset(
                    torch.from_numpy(X_train).float(),
                    torch.from_numpy(y_train).float()
                )
                train_loader = DataLoader(mlp_ds, batch_size=32, shuffle=True)
                clf.fit_with_loader(train_loader, epochs=50)
                y_proba_all['MLP'].append(clf.predict_proba(X_test))
            else:
                clf.fit(X_train, y_train)
                y_proba = clf.predict_proba(X_test)
                if isinstance(y_proba, list):
                    y_proba = np.array([p[:, 1] for p in y_proba]).T
                y_proba_all[name].append(y_proba)

    return y_true_all, y_proba_all

def linear_probe_online(csv_data : str, n_split=5, random_state=42,balance : bool = False,encoder_name : str = 'perch2'):
    #0. Initialise dataset 
    df = pd.read_csv(csv_data)
    label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
    # 1. Initialize the split
    kf = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
    clf_names = ['SVM', 'Random Forest', 'MLP']

    # 2. Setup storage
    y_true_all = [] # Will hold one y_test per fold (length = n_split)
    y_proba_all = {name: [] for name in clf_names}

    #3. Extract Encoder 
    encoder = extract_encoder(encoder_name, device='cpu')

    for fold, (train_idx, test_idx) in enumerate(kf.split(df, df[label_cols].values)):
        print(f"Processing Fold {fold+1}...")

        # 1. Instantiate Datasets for this fold
        train_ds = PipistrelleDataset(df.iloc[train_idx], is_training=True, resample=True)
        test_ds = PipistrelleDataset(df.iloc[test_idx], is_training=False, resample=False)

        # 2. Extract Static Features for SVM/RF (One pass through the dataset)
        # This gives SVM/RF one augmented version of the training data
        X_train_static, y_train_static = build_feature_bank(train_ds, encoder, encoder_name, device='cpu')
        X_test_static, y_test_static = build_feature_bank(test_ds, encoder, encoder_name, device='cpu')

        # 3. Scaler (Fit on training snapshot)
        scaler = StandardScaler()
        X_train_static = scaler.fit_transform(X_train_static)
        X_test_static = scaler.transform(X_test_static)
        
        y_true_all.append(y_test_static)

        models = {
            'SVM': OneVsRestClassifier(SVC(
                probability=True, 
                random_state=random_state,
                class_weight='balanced' if balance else None)),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                class_weight='balanced' if balance else None), # RF is natively multi-label
             #Random Forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state)),
            'MLP': BalancedMLP(
                input_dim=X_train_static.shape[1],
                hidden_dim=128,
                lr=0.001,
                epochs=50,
                dropout=0.2,
                balanced=balance,
                batch_norm=True
            )
        }

        for name, clf in models.items():
            if name == 'MLP':
                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                clf.fit_with_loader(train_loader, epochs=50) # Custom fit method
                y_proba_all['MLP'].append(clf.predict_proba(X_test_static))
            else :
                clf.fit(X_train_static, y_train_static)
                y_proba = clf.predict_proba(X_test_static)
            
                 # predict_proba for multi-label often returns a list of arrays
                # We want to ensure it's a consistent [Samples, Labels] array
                if isinstance(y_proba, list):
                    # Convert list of [Samples, 2] to [Samples, Labels] using the positive class proba
                    y_proba = np.array([p[:, 1] for p in y_proba]).T
            
                y_proba_all[name].append(y_proba)
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return y_true_all, y_proba_all


def linear_probe(X, y, n_split=5, random_state=42,balance : bool = False,oversample:bool = False):
    # 1. Initialize the split
    kf = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
    clf_names = ['SVM', 'Random Forest', 'MLP','Random Guesser']

    # 2. Setup storage
    y_true_all = [] # Will hold one y_test per fold (length = n_split)
    y_proba_all = {name: [] for name in clf_names}

    if oversample : aug_pipeline = AugmentationPipeline() 

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"Processing Fold {fold+1}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if oversample : X_train, y_train =aug_pipeline.iterative_oversample(X_train, y_train)
        # 3. CRITICAL: Scale features for SVM and MLP
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Store y_test ONCE per fold
        y_true_all.append(y_test)

        models = {
            'SVM': OneVsRestClassifier(SVC(
                probability=True, 
                random_state=random_state,
                class_weight='balanced' if balance else None)),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                class_weight='balanced' if balance else None), # RF is natively multi-label
            
            #'Random Forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state)),
            #'MLP' : OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state))
            
            'MLP': BalancedMLP(
                input_dim=X.shape[1],
                hidden_dim=128,
                lr=0.001,
                epochs=50,
                dropout=0.2,
                balanced=balance,
                batch_norm=False
            ),
            'Random Guesser' : MultilabelPrevalenceBaseline(type='stochastic')
        }

        for name, clf in models.items():
            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)
            
            # predict_proba for multi-label often returns a list of arrays
            # We want to ensure it's a consistent [Samples, Labels] array
            if isinstance(y_proba, list):
                # Convert list of [Samples, 2] to [Samples, Labels] using the positive class proba
                y_proba = np.array([p[:, 1] for p in y_proba]).T
            
            y_proba_all[name].append(y_proba)
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return y_true_all, y_proba_all

def linear_probe_tuned(X, y, n_split_out=5,n_split_in=5, num_trials=5,random_state=42,balance : bool = False):
    # 1. Initialize the split
    scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
    all_results = []

    for i in range(num_trials) :
        trial_seed = random_state + i
        print(f"Starting Trial {i+1}/{num_trials} with random_state={trial_seed}...")

        #set random seeds
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)

        #instantiate model configurations
        model_params = {
            'SVM': {
                'model': OneVsRestClassifier(SVC(
                            probability=True, 
                            random_state=trial_seed, # Updated dynamically
                            class_weight='balanced' if balance else None)),
                'params': {
                    'model__estimator__C': [1, 10, 20],
                    'model__estimator__kernel': ['rbf', 'linear'],
                    'model__estimator__gamma': ['scale', 'auto', 0.01, 0.1]
                }
            }, 
            'Logistic Regression': {
                'model': OneVsRestClassifier(LogisticRegression(
                            max_iter=1000,             
                            random_state=trial_seed,     
                            class_weight='balanced' if balance else None
                         )),
                'params': {
                    'model__estimator__C': [0.1, 1.0, 10.0],       # Standard inverse regularization strengths
                    'model__estimator__solver': ['lbfgs']   #  multi-label optimization solvers
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                            n_estimators=100, 
                            random_state=trial_seed, # Updated dynamically
                            class_weight='balanced' if balance else None), 
                'params': {
                    'model__n_estimators': [100],
                    'model__max_depth': [None, 10, 20]
                }
            },
            'MLP' : {
                'model' : BalancedMLP(
                    input_dim=X.shape[1],
                    hidden_dim=128,
                    lr=0.001,
                    epochs=50,
                    dropout=0.2,
                    balanced=balance,
                    batch_norm=False,
                    random_state=trial_seed
                ),
                'params' : {
                    'model__lr':[0.001],
                    'model__hidden_dim':[128],
                    'model__epochs':[50],
                    'model__dropout':[0.2, 0.5]
                }
            },
            'Prevalence guesser': {
                # Ensure your stochastic baseline uses NumPy or accepts a random_state seed!
                #'model': MultilabelPrevalenceBaseline(type='stochastic', random_state=trial_seed),
                'model' : DummyClassifier(strategy='stratified', random_state=trial_seed),
                'params': {}
            }
        }

        #Cross validation techniques for inner and outer loop
        inner_cv = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=trial_seed)
        outer_cv = MultilabelStratifiedKFold(n_splits=n_split_out, shuffle=True, random_state=trial_seed)

        #Nester CV with parameter optimisation for each model
        for model_name, mp in model_params.items():
            print(f"  Tuning and evaluating model: {model_name}")
            all_y_true = []
            all_y_pred_proba = []
            all_test_indices = []

            outer_scores = []

            for fold ,(train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                print(f"    Evaluating fold {fold+1}/{n_split_out}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', mp['model'])
                ])

                clf = GridSearchCV(estimator=pipeline,param_grid=mp['params'],cv=inner_cv,
                                   scoring=scorer,refit=True,n_jobs=1 if model_name == 'MLP' else -1)

                # fit on outer-train
                clf.fit(X_train, y_train)

                # predict on outer-test
                y_pred_proba = clf.predict_proba(X_test)

                #checking for y_pred_proba format and converting to [Samples, Labels] if needed
                if isinstance(y_pred_proba, list):
                    # For a list of arrays, extract the positive probability (column index 1) for each class
                    y_pred_proba = np.column_stack([prob[:, 1] for prob in y_pred_proba])
                elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 3:
                    # Alternative 3D representation sometimes returned by multi-output setups
                    y_pred_proba = y_pred_proba[:, :, 1].T

                # fold score
                fold_score = average_precision_score(y_test,y_pred_proba,average='macro')
                outer_scores.append(fold_score)

                # ---------------------------------
                # STORE OOF PREDICTIONS
                # ---------------------------------
                all_y_true.append(y_test)
                all_y_pred_proba.append(y_pred_proba)
                all_test_indices.append(test_idx)
            
            #Concatenate fold results to get out of fold predictions
            y_true_cv = all_y_true
            y_pred_proba_cv = all_y_pred_proba
            all_y_true = np.concatenate(all_y_true, axis=0)
            all_y_pred_proba = np.concatenate(all_y_pred_proba, axis=0)
            all_test_indices = np.concatenate(all_test_indices, axis=0)
            all_results.append({
                'trial': i,
                'model': model_name,

                'mean_AP': np.mean(outer_scores),
                'std_AP': np.std(outer_scores, ddof=1),

                'y_true_cv' : y_true_cv,
                'y_pred_proba_cv' : y_pred_proba_cv,
                'oof_y_true': all_y_true,
                'oof_y_pred_proba': all_y_pred_proba,
                'oof_indices': all_test_indices
            })
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return all_results

