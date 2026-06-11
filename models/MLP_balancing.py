



from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from models.linear_probes import BalancedMLP, MultilabelPrevalenceBaseline
from models.focal_loss import FocalLoss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import random

from preprocessing.dataset import AugmentationPipeline

def balancing_mlp(X, y, n_split_out=5,n_split_in=5, num_trials=5,random_state=42,balance : bool = False):
    # 1. Initialize the split
    scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
    all_results = []

    for i in range(num_trials) :
        print(f"Starting Trial {i+1}/{num_trials} with random_state={random_state + i}...")
        model_params = {
            'MLP_Baseline': {
                'model': BalancedMLP(input_dim=X.shape[1],balanced=False),
                'params': {} # Plain Binary Cross Entropy
            },
            'MLP_ClassWeights': {
                'model': BalancedMLP(input_dim=X.shape[1],balanced=True),
                'params': {} # Scaled BCE loss based on minority prevalence
            },
            'MLP_FocalLoss': {
                'model': BalancedMLP(input_dim=X.shape[1],focal_loss=True),
                'params': {
                    'model__focal_gamma': [1.0, 2.0, 5.0], # Higher gamma = focus more on hard bat calls
                    'model__focal_alpha': [0.25, 0.5, 0.75]
                }
            },
            'MLP_Oversampled': {
                # Use standard BCE but feed it data passed through an oversampler in your pipeline
                'model': BalancedMLP(input_dim=X.shape[1],balanced=False),
                'params': {}
            }
        }
        #Cross validation techniques for inner and outer loop
        inner_cv = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=i)
        outer_cv = MultilabelStratifiedKFold(n_splits=n_split_out, shuffle=True, random_state=i)

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

                if model_name == 'MLP_Oversampled' :
                    clf = mp['model']
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    X_train,y_train = AugmentationPipeline().iterative_oversample(X_train,y_train,random_state=42+i)
                    clf.fit(X_train, y_train)
                else :
                    clf = GridSearchCV(estimator=pipeline,param_grid=mp['params'],cv=inner_cv,
                                   scoring=scorer,refit=True,n_jobs=-1)
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
            all_y_true = np.concatenate(all_y_true, axis=0)
            all_y_pred_proba = np.concatenate(all_y_pred_proba, axis=0)
            all_test_indices = np.concatenate(all_test_indices, axis=0)
            all_results.append({
                'trial': i,
                'model': model_name,

                'mean_AP': np.mean(outer_scores),
                'std_AP': np.std(outer_scores, ddof=1),

                'oof_y_true': all_y_true,
                'oof_y_pred_proba': all_y_pred_proba,
                'oof_indices': all_test_indices
            })
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return all_results


def balancing_mlp_val(X, y, n_split_out=5,n_split_in=5, num_trials=5,random_state=42):
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
            'MLP_Baseline': {
                'model': BalancedMLP(input_dim=X.shape[1],balanced=False,random_state=trial_seed),
                'params': {} # Plain Binary Cross Entropy
            },
            'MLP_ClassWeights': {
                'model': BalancedMLP(input_dim=X.shape[1],balanced=True,random_state=trial_seed),
                'params': {} # Scaled BCE loss based on minority prevalence
            },
            'MLP_FocalLoss': {
                'model': BalancedMLP(input_dim=X.shape[1],focal_loss=True,random_state=trial_seed),
                'params': {
                    'model__focal_gamma': [1.0, 2.0, 5.0], # Higher gamma = focus more on hard bat calls
                    'model__focal_alpha': [0.25, 0.5, 0.75]
                }
            },
            'MLP_Oversampled': {
                # Use standard BCE but feed it data passed through an oversampler in your pipeline
                'model': BalancedMLP(input_dim=X.shape[1],balanced=False,random_state=trial_seed),
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

            fold_train_histories = []
            fold_val_histories = []

            for fold ,(train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                print(f"    Evaluating fold {fold+1}/{n_split_out}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', mp['model'])
                ])

                if model_name == 'MLP_Oversampled' :
                    clf = mp['model']
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    X_train,y_train = AugmentationPipeline().iterative_oversample(X_train,y_train,random_state=trial_seed,target_percentage=0.5)
                    clf.fit(X_train, y_train,X_val = X_test,y_val = y_test)
                    fold_train_histories.append(clf.train_loss_history_)
                    fold_val_histories.append(clf.val_loss_history_)
                elif model_name == 'MLP_FocalLoss':
                    clf = GridSearchCV(estimator=pipeline,param_grid=mp['params'],cv=inner_cv,
                                   scoring=scorer,refit=True,n_jobs=-1)
                    # fit on outer-train
                    clf.fit(X_train, y_train,model__X_val=X_test, model__y_val=y_test)
                    best_model = clf.best_estimator_.named_steps['model']
                    fold_train_histories.append(best_model.train_loss_history_)
                    fold_val_histories.append(best_model.val_loss_history_)
                else :
                    clf = mp['model']
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    clf.fit(X_train, y_train,X_val = X_test,y_val = y_test)
                    fold_train_histories.append(clf.train_loss_history_)
                    fold_val_histories.append(clf.val_loss_history_)

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
                'oof_indices': all_test_indices,
                'train_histories': fold_train_histories,
                'val_histories': fold_val_histories
            })
    
    # Return as numpy arrays for easier use in your compute_cv_stats
    return all_results

import pandas as pd
from models.feature_generation import extract_encoder,pool_features,build_feature_bank
from preprocessing.dataset import PipistrelleDataset
import torch 
from torch.utils.data import DataLoader, TensorDataset


def data_augmented_mlp(csv_data,root_dir,X_perch,y_perch, n_split=5, num_trials=5,random_state=42):
    # 1. Initialize the split
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(csv_data)
    all_results = []
    encoder_name = 'perch2'
    encoder = extract_encoder(encoder_name, device='cpu')
    label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']

    for i in range(num_trials) :
        trial_seed = random_state + i
        print(f"Starting Trial {i+1}/{num_trials} with random_state={trial_seed}...")

        #set random seeds
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)

        model_params = {
            'MLP_Baseline': {
                'model': BalancedMLP(input_dim=1536,balanced=False,random_state=trial_seed),
                'params': {} # Plain Binary Cross Entropy
            }
        }
        #Cross validation techniques for inner and outer loop
        cv = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=trial_seed)

        #Nester CV with parameter optimisation for each model
        for model_name, mp in model_params.items():
            print(f"  Tuning and evaluating model: {model_name}")
            all_y_true = []
            all_y_pred_proba = []
            all_test_indices = []

            outer_scores = []

            for fold ,(train_idx, test_idx) in enumerate(cv.split(df, df[label_cols].values)):
                print(f"    Evaluating fold {fold+1}/{n_split}")
                #Train and test datasets
                train_ds = PipistrelleDataset(data_input = df.iloc[train_idx],
                                              root_dir = root_dir,
                                              is_training=True, 
                                              resample=True,
                                              time_shift=True, 
                                              encoder=encoder_name)
                #test_ds  = PipistrelleDataset(data_input =df.iloc[test_idx], 
                #                              root_dir = root_dir, 
                #                              is_training=False,
                #                              resample=False,
                #                              encoder=encoder_name)
                #Build train features and mean pool
                feature_list_tr, y_train = build_feature_bank(train_ds, encoder, encoder_name, device=device)
                print("Train features extracted")
                X_train = pool_features(
                pool_features(feature_list_tr, windows=True, method='mean',encoder=encoder_name),
                windows=False, window_pooled=True, method='mean',encoder=encoder_name)

                #Build test features and mean pool
                #feature_list_ts,y_test = build_feature_bank(test_ds,  encoder, encoder_name, device=device)
                #print("Test features extracted")
                #X_test = pool_features(
                #pool_features(feature_list_ts, windows=True, method='mean',encoder=encoder_name),
                #windows=False, window_pooled=True, method='mean',encoder=encoder_name)
                X_test = X_perch[test_idx]
                y_test = y_perch[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

                clf = mp['model']
                clf.fit(X_train,y_train)
                print("Model trained")
                
                # predict on cv test
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
