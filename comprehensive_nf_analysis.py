# -*- coding: utf-8 -*-
"""
Comprehensive Nanofiltration Membrane Machine Learning Analysis System
综合纳滤膜机器学习分析系统

Created for predicting membrane permeance and Li-Mg separation coefficient
用于预测膜渗透性和锂镁分离系数

@author: ML Analysis System
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from datetime import datetime

# Machine Learning Libraries / 机器学习库
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

# SHAP for model interpretability / SHAP用于模型可解释性
import shap

# Progress tracking / 进度跟踪
from tqdm import tqdm

# Suppress warnings for cleaner output / 抑制警告以获得更清洁的输出
warnings.filterwarnings('ignore')

class NanofiltrationMLAnalyzer:
    """
    Comprehensive ML Analysis System for Nanofiltration Membrane Data
    纳滤膜数据综合机器学习分析系统
    """
    
    def __init__(self, data_file='Data.xlsx', output_dir=None):
        """
        Initialize the analyzer / 初始化分析器
        
        Parameters:
        - data_file: Path to the data file / 数据文件路径
        - output_dir: Output directory for results / 结果输出目录
        """
        self.data_file = data_file
        self.output_dir = output_dir or os.path.expanduser('~/Desktop/NF_Analysis_Results')
        self.create_output_directories()
        
        # Target variables / 目标变量
        self.target_variables = {
            'permeance': 'Permeance',
            'separation': 'Lithium - Magnesium Separation Coefficient'
        }
        
        # Results storage / 结果存储
        self.results = {}
        self.models = {}
        self.processed_data = {}
        
        # Model configurations / 模型配置
        self.n_experiments = 300  # Following example code pattern / 遵循示例代码模式
        
        print(f"🔬 Nanofiltration ML Analyzer Initialized")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"🎯 Target Variables: {list(self.target_variables.keys())}")
    
    def create_output_directories(self):
        """Create organized output directories / 创建有组织的输出目录"""
        subdirs = ['csv_results', 'figures', 'shap_analysis', 'model_comparisons']
        
        for subdir in subdirs:
            path = os.path.join(self.output_dir, subdir)
            os.makedirs(path, exist_ok=True)
        
        print(f"📂 Created output directories in: {self.output_dir}")
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the nanofiltration membrane data
        加载和预处理纳滤膜数据
        """
        print("\n🔄 Loading and preprocessing data...")
        
        # Load data / 加载数据
        df = pd.read_excel(self.data_file)
        print(f"📊 Original data shape: {df.shape}")
        
        # Remove the units row (first row) / 移除单位行（第一行）
        df = df.iloc[1:].reset_index(drop=True)
        
        # Clean column names / 清理列名
        df.columns = df.columns.str.strip()
        
        # Identify feature columns (exclude non-numeric and target columns) / 识别特征列
        exclude_cols = ['DataNo.', 'Monomers', 'Support Membrane'] + list(self.target_variables.values())
        
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                feature_cols.append(col)
        
        print(f"🎯 Feature columns identified: {len(feature_cols)}")
        print(f"📝 Features: {feature_cols}")
        
        # Process each target variable / 处理每个目标变量
        for target_name, target_col in self.target_variables.items():
            print(f"\n🎯 Processing target: {target_name} ({target_col})")
            
            # Prepare feature and target data / 准备特征和目标数据
            X_raw = df[feature_cols].copy()
            y_raw = df[target_col].copy()
            
            # Clean and convert data / 清理和转换数据
            X_clean = self._clean_features(X_raw)
            y_clean = self._clean_target(y_raw)
            
            # Remove rows with missing target values / 移除目标值缺失的行
            valid_indices = ~pd.isna(y_clean)
            X_final = X_clean[valid_indices]
            y_final = y_clean[valid_indices]
            
            print(f"✅ Final dataset shape: X={X_final.shape}, y={y_final.shape}")
            
            # Store processed data / 存储处理后的数据
            self.processed_data[target_name] = {
                'X': X_final,
                'y': y_final,
                'feature_names': feature_cols,
                'original_size': len(df),
                'final_size': len(y_final)
            }
    
    def _clean_features(self, X):
        """Clean feature data / 清理特征数据"""
        X_clean = X.copy()
        
        for col in X_clean.columns:
            # Replace '/' and other non-numeric values with NaN / 将'/'等非数值替换为NaN
            X_clean[col] = X_clean[col].replace(['/', '\\', 'NaN', 'nan', ''], np.nan)
            
            # Convert to numeric / 转换为数值型
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        
        # Impute missing values using median / 使用中位数填充缺失值
        imputer = SimpleImputer(strategy='median')
        X_clean = pd.DataFrame(
            imputer.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        return X_clean
    
    def _clean_target(self, y):
        """Clean target data / 清理目标数据"""
        y_clean = y.copy()
        
        # Replace non-numeric values with NaN / 将非数值替换为NaN
        y_clean = y_clean.replace(['/', '\\', 'NaN', 'nan', ''], np.nan)
        
        # Convert to numeric / 转换为数值型
        y_clean = pd.to_numeric(y_clean, errors='coerce')
        
        return y_clean
    
    def run_comprehensive_analysis(self):
        """
        Run comprehensive ML analysis for all targets and models
        对所有目标和模型运行综合ML分析
        """
        print("\n🚀 Starting Comprehensive ML Analysis...")
        
        # Load and preprocess data / 加载和预处理数据
        self.load_and_preprocess_data()
        
        # Analyze each target variable / 分析每个目标变量
        for target_name in self.target_variables.keys():
            print(f"\n{'='*60}")
            print(f"🎯 ANALYZING TARGET: {target_name.upper()}")
            print(f"{'='*60}")
            
            self.analyze_target(target_name)
        
        # Generate comparative analysis / 生成比较分析
        self.generate_model_comparisons()
        
        print("\n🎉 Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
    
    def analyze_target(self, target_name):
        """
        Analyze a specific target variable with all models
        使用所有模型分析特定目标变量
        """
        data = self.processed_data[target_name]
        X, y = data['X'], data['y']
        feature_names = data['feature_names']
        
        # Initialize results storage for this target / 初始化此目标的结果存储
        self.results[target_name] = {}
        self.models[target_name] = {}
        
        # Define models / 定义模型
        models_config = {
            'XGBoost': {
                'model_class': xgb.XGBRegressor,
                'params': {
                    'n_estimators': 180,
                    'learning_rate': 0.06,
                    'max_depth': 6,
                    'objective': 'reg:squarederror',
                    'random_state': 7
                }
            },
            'RandomForest': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': 180,
                    'max_depth': 10,
                    'random_state': 7,
                    'n_jobs': -1
                }
            },
            'NeuralNetwork': {
                'model_class': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': (100, 50),
                    'max_iter': 1000,
                    'random_state': 7,
                    'alpha': 0.001
                }
            }
        }
        
        # Analyze each model / 分析每个模型
        for model_name, config in models_config.items():
            print(f"\n🔍 Training {model_name} model...")
            self.train_and_evaluate_model(target_name, model_name, config, X, y, feature_names)
        
        # Generate SHAP analysis for best model / 为最佳模型生成SHAP分析
        self.generate_shap_analysis(target_name, X, feature_names)
    
    def train_and_evaluate_model(self, target_name, model_name, config, X, y, feature_names):
        """
        Train and evaluate a model with cross-validation
        使用交叉验证训练和评估模型
        """
        # Storage for multiple experiments / 多次实验的存储
        experiment_results = []
        feature_importances = []
        best_model = None
        best_score = -np.inf
        
        # Standardize features for neural networks / 为神经网络标准化特征
        if model_name == 'NeuralNetwork':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
            scaler = None
        
        print(f"🔄 Running {self.n_experiments} experiments...")
        
        # Multiple experiments for robust evaluation / 多次实验以进行稳健评估
        for i in tqdm(range(self.n_experiments), desc=f"Training {model_name}"):
            try:
                # Split data / 分割数据
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=i
                )
                
                # Create and train model / 创建和训练模型
                model = config['model_class'](**config['params'])
                model.fit(X_train, y_train)
                
                # Make predictions / 进行预测
                y_pred = model.predict(X_test)
                
                # Calculate metrics / 计算指标
                pearson_r = stats.pearsonr(y_test, y_pred)[0]
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Store results / 存储结果
                result = {
                    'experiment': i,
                    'pearson_r': pearson_r,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                }
                
                # Extract feature importance / 提取特征重要性
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    result.update({f'importance_{j}': importance[j] for j in range(len(importance))})
                    feature_importances.append(importance)
                
                experiment_results.append(result)
                
                # Track best model / 跟踪最佳模型
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    
            except Exception as e:
                print(f"⚠️ Experiment {i} failed: {str(e)}")
                continue
        
        # Store results / 存储结果
        self.results[target_name][model_name] = {
            'experiments': experiment_results,
            'feature_names': feature_names,
            'scaler': scaler
        }
        
        self.models[target_name][model_name] = best_model
        
        # Calculate summary statistics / 计算汇总统计
        self.calculate_model_statistics(target_name, model_name)
        
        # Save detailed results to CSV / 将详细结果保存为CSV
        self.save_model_results_csv(target_name, model_name)
        
        # Generate visualizations / 生成可视化
        self.generate_model_visualizations(target_name, model_name)
    
    def calculate_model_statistics(self, target_name, model_name):
        """Calculate summary statistics for model performance / 计算模型性能的汇总统计"""
        results = self.results[target_name][model_name]['experiments']
        df_results = pd.DataFrame(results)
        
        # Calculate mean and std for each metric / 计算每个指标的均值和标准差
        metrics = ['pearson_r', 'r2', 'rmse', 'mae']
        stats_summary = {}
        
        for metric in metrics:
            stats_summary[f'{metric}_mean'] = df_results[metric].mean()
            stats_summary[f'{metric}_std'] = df_results[metric].std()
        
        # Calculate feature importance statistics / 计算特征重要性统计
        importance_cols = [col for col in df_results.columns if col.startswith('importance_')]
        if importance_cols:
            feature_importance_stats = {}
            for i, feature_name in enumerate(self.results[target_name][model_name]['feature_names']):
                importance_col = f'importance_{i}'
                if importance_col in df_results.columns:
                    feature_importance_stats[feature_name] = {
                        'mean': df_results[importance_col].mean(),
                        'std': df_results[importance_col].std()
                    }
            
            stats_summary['feature_importance'] = feature_importance_stats
        
        self.results[target_name][model_name]['statistics'] = stats_summary
        
        # Print summary / 打印摘要
        print(f"\n📊 {model_name} Results Summary:")
        for metric in metrics:
            mean_val = stats_summary[f'{metric}_mean']
            std_val = stats_summary[f'{metric}_std']
            print(f"   {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    def save_model_results_csv(self, target_name, model_name):
        """Save detailed model results to CSV / 将详细模型结果保存为CSV"""
        results = self.results[target_name][model_name]['experiments']
        df_results = pd.DataFrame(results)
        
        # Save experiment results / 保存实验结果
        filename = f"{target_name}_{model_name}_detailed_results.csv"
        filepath = os.path.join(self.output_dir, 'csv_results', filename)
        df_results.to_csv(filepath, index=False, encoding='utf-8')
        
        # Save summary statistics / 保存汇总统计
        stats = self.results[target_name][model_name]['statistics']
        
        summary_data = []
        metrics = ['pearson_r', 'r2', 'rmse', 'mae']
        for metric in metrics:
            summary_data.append({
                'Metric': metric.upper(),
                'Mean': stats[f'{metric}_mean'],
                'Std': stats[f'{metric}_std']
            })
        
        df_summary = pd.DataFrame(summary_data)
        summary_filename = f"{target_name}_{model_name}_performance_summary.csv"
        summary_filepath = os.path.join(self.output_dir, 'csv_results', summary_filename)
        df_summary.to_csv(summary_filepath, index=False, encoding='utf-8')
        
        # Save feature importance if available / 如果可用，保存特征重要性
        if 'feature_importance' in stats:
            importance_data = []
            for feature, imp_stats in stats['feature_importance'].items():
                importance_data.append({
                    'Feature': feature,
                    'Importance_Mean': imp_stats['mean'],
                    'Importance_Std': imp_stats['std']
                })
            
            if importance_data:
                df_importance = pd.DataFrame(importance_data)
                df_importance = df_importance.sort_values('Importance_Mean', ascending=False)
                
                importance_filename = f"{target_name}_{model_name}_feature_importance.csv"
                importance_filepath = os.path.join(self.output_dir, 'csv_results', importance_filename)
                df_importance.to_csv(importance_filepath, index=False, encoding='utf-8')
        
        print(f"💾 Saved results to: {filepath}")
    
    def generate_model_visualizations(self, target_name, model_name):
        """Generate visualizations for model performance / 为模型性能生成可视化"""
        
        # Set up plotting style / 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Font settings for Chinese and English / 中英文字体设置
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['figure.dpi'] = 100
        
        results = self.results[target_name][model_name]['experiments']
        df_results = pd.DataFrame(results)
        
        # 1. Performance metrics distribution / 性能指标分布
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - {target_name.title()} Performance Distribution\n'
                    f'{model_name} - {target_name.title()}性能分布', fontsize=16, fontweight='bold')
        
        metrics = ['pearson_r', 'r2', 'rmse', 'mae']
        metric_labels = ['Pearson Correlation', 'R² Score', 'RMSE', 'MAE']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx//2, idx%2]
            ax.hist(df_results[metric], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{label} Distribution')
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency / 频率')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text / 添加统计文本
            mean_val = df_results[metric].mean()
            std_val = df_results[metric].std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
            ax.text(0.05, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        filename = f"{target_name}_{model_name}_performance_distribution.png"
        filepath = os.path.join(self.output_dir, 'figures', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance visualization (if available) / 特征重要性可视化（如果可用）
        stats = self.results[target_name][model_name]['statistics']
        if 'feature_importance' in stats:
            self.plot_feature_importance(target_name, model_name, stats['feature_importance'])
        
        # 3. Model prediction scatter plot / 模型预测散点图
        self.plot_prediction_scatter(target_name, model_name)
    
    def plot_feature_importance(self, target_name, model_name, feature_importance):
        """Plot feature importance / 绘制特征重要性"""
        
        # Prepare data for plotting / 准备绘图数据
        features = list(feature_importance.keys())
        importance_means = [feature_importance[f]['mean'] for f in features]
        importance_stds = [feature_importance[f]['std'] for f in features]
        
        # Sort by importance / 按重要性排序
        sorted_indices = np.argsort(importance_means)[::-1]
        features_sorted = [features[i] for i in sorted_indices]
        means_sorted = [importance_means[i] for i in sorted_indices]
        stds_sorted = [importance_stds[i] for i in sorted_indices]
        
        # Create plot / 创建图表
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(features_sorted)), means_sorted, 
                      yerr=stds_sorted, capsize=5, alpha=0.7)
        
        plt.title(f'{model_name} - {target_name.title()} Feature Importance\n'
                 f'{model_name} - {target_name.title()}特征重要性', fontsize=16, fontweight='bold')
        plt.xlabel('Features / 特征', fontsize=12)
        plt.ylabel('Importance / 重要性', fontsize=12)
        
        # Rotate x-axis labels for better readability / 旋转x轴标签以提高可读性
        plt.xticks(range(len(features_sorted)), features_sorted, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars / 在条形图上添加数值标签
        for i, (mean, std) in enumerate(zip(means_sorted, stds_sorted)):
            plt.text(i, mean + std, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        filename = f"{target_name}_{model_name}_feature_importance.png"
        filepath = os.path.join(self.output_dir, 'figures', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_scatter(self, target_name, model_name):
        """Create prediction vs actual scatter plot / 创建预测与实际的散点图"""
        
        # Get best model and data / 获取最佳模型和数据
        model = self.models[target_name][model_name]
        data = self.processed_data[target_name]
        X, y = data['X'], data['y']
        
        # Apply scaling if needed / 如果需要应用缩放
        scaler = self.results[target_name][model_name].get('scaler')
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions on full dataset / 对完整数据集进行预测
        y_pred = model.predict(X_scaled)
        
        # Calculate metrics / 计算指标
        pearson_r = stats.pearsonr(y, y_pred)[0]
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Create plot / 创建图表
        plt.figure(figsize=(10, 8))
        plt.scatter(y, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line / 完美预测线
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.title(f'{model_name} - {target_name.title()} Prediction vs Actual\n'
                 f'{model_name} - {target_name.title()}预测值与实际值', fontsize=16, fontweight='bold')
        plt.xlabel('Actual Values / 实际值', fontsize=12)
        plt.ylabel('Predicted Values / 预测值', fontsize=12)
        
        # Add statistics text / 添加统计文本
        stats_text = f'Pearson r: {pearson_r:.4f}\nR²: {r2:.4f}\nRMSE: {rmse:.4f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        filename = f"{target_name}_{model_name}_prediction_scatter.png"
        filepath = os.path.join(self.output_dir, 'figures', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_shap_analysis(self, target_name, X, feature_names):
        """Generate SHAP analysis for the best performing model / 为表现最佳的模型生成SHAP分析"""
        
        print(f"🔍 Generating SHAP analysis for {target_name}...")
        
        # Find best model based on R² score / 基于R²分数找到最佳模型
        best_model_name = None
        best_r2 = -np.inf
        
        for model_name in self.results[target_name].keys():
            stats = self.results[target_name][model_name]['statistics']
            r2_mean = stats['r2_mean']
            if r2_mean > best_r2:
                best_r2 = r2_mean
                best_model_name = model_name
        
        if best_model_name is None:
            print("⚠️ No valid model found for SHAP analysis")
            return
        
        print(f"🏆 Best model: {best_model_name} (R² = {best_r2:.4f})")
        
        # Get best model and prepare data / 获取最佳模型并准备数据
        model = self.models[target_name][best_model_name]
        scaler = self.results[target_name][best_model_name].get('scaler')
        
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        try:
            # Create SHAP explainer / 创建SHAP解释器
            if best_model_name == 'XGBoost':
                explainer = shap.Explainer(model)
                shap_values = explainer(X_scaled)
            elif best_model_name == 'RandomForest':
                # Use TreeExplainer for RandomForest with a sample of background data
                background_data = X_scaled[:min(100, len(X_scaled))]
                explainer = shap.TreeExplainer(model, background_data)
                shap_values = explainer.shap_values(X_scaled)
                # Convert to SHAP values object for consistency
                shap_values = shap.Explanation(values=shap_values, feature_names=feature_names)
            else:  # Neural Network
                explainer = shap.Explainer(model, X_scaled[:min(100, len(X_scaled))])
                shap_values = explainer(X_scaled)
            
            # Save SHAP values to CSV / 将SHAP值保存为CSV
            shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
            shap_filename = f"{target_name}_shap_values.csv"
            shap_filepath = os.path.join(self.output_dir, 'shap_analysis', shap_filename)
            shap_df.to_csv(shap_filepath, index=False, encoding='utf-8')
            
            # Generate SHAP visualizations / 生成SHAP可视化
            self.create_shap_visualizations(target_name, best_model_name, shap_values, feature_names)
            
            print(f"💾 SHAP analysis saved for {target_name}")
            
        except Exception as e:
            print(f"⚠️ SHAP analysis failed for {target_name}: {str(e)}")
    
    def create_shap_visualizations(self, target_name, model_name, shap_values, feature_names):
        """Create SHAP visualizations / 创建SHAP可视化"""
        
        # 1. Feature importance plot / 特征重要性图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name} ({target_name.title()})\n'
                 f'SHAP特征重要性 - {model_name} ({target_name.title()})', fontsize=14)
        plt.tight_layout()
        
        filename = f"{target_name}_shap_feature_importance.png"
        filepath = os.path.join(self.output_dir, 'shap_analysis', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Summary plot / 汇总图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {model_name} ({target_name.title()})\n'
                 f'SHAP汇总图 - {model_name} ({target_name.title()})', fontsize=14)
        plt.tight_layout()
        
        filename = f"{target_name}_shap_summary.png"
        filepath = os.path.join(self.output_dir, 'shap_analysis', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_model_comparisons(self):
        """Generate comprehensive model comparison visualizations / 生成综合模型比较可视化"""
        
        print("\n📊 Generating model comparisons...")
        
        for target_name in self.target_variables.keys():
            self.create_target_model_comparison(target_name)
        
        # Create overall comparison / 创建总体比较
        self.create_overall_comparison()
    
    def create_target_model_comparison(self, target_name):
        """Create model comparison for a specific target / 为特定目标创建模型比较"""
        
        models_in_target = list(self.results[target_name].keys())
        metrics = ['pearson_r', 'r2', 'rmse', 'mae']
        
        # Prepare comparison data / 准备比较数据
        comparison_data = []
        for model_name in models_in_target:
            stats = self.results[target_name][model_name]['statistics']
            row = {'Model': model_name}
            for metric in metrics:
                row[f'{metric}_mean'] = stats[f'{metric}_mean']
                row[f'{metric}_std'] = stats[f'{metric}_std']
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save comparison CSV / 保存比较CSV
        filename = f"{target_name}_model_comparison.csv"
        filepath = os.path.join(self.output_dir, 'model_comparisons', filename)
        df_comparison.to_csv(filepath, index=False, encoding='utf-8')
        
        # Create comparison visualization / 创建比较可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Comparison - {target_name.title()}\n'
                    f'模型比较 - {target_name.title()}', fontsize=16, fontweight='bold')
        
        metric_labels = ['Pearson Correlation', 'R² Score', 'RMSE', 'MAE']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx//2, idx%2]
            
            means = df_comparison[f'{metric}_mean']
            stds = df_comparison[f'{metric}_std']
            
            bars = ax.bar(df_comparison['Model'], means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f'{label} Comparison')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            
            # Add value labels / 添加数值标签
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std, f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        filename = f"{target_name}_model_comparison.png"
        filepath = os.path.join(self.output_dir, 'model_comparisons', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_overall_comparison(self):
        """Create overall comparison across all targets and models / 创建所有目标和模型的总体比较"""
        
        # Prepare overall comparison data / 准备总体比较数据
        overall_data = []
        
        for target_name in self.target_variables.keys():
            for model_name in self.results[target_name].keys():
                stats = self.results[target_name][model_name]['statistics']
                row = {
                    'Target': target_name.title(),
                    'Model': model_name,
                    'Pearson_R': stats['pearson_r_mean'],
                    'R2_Score': stats['r2_mean'],
                    'RMSE': stats['rmse_mean'],
                    'MAE': stats['mae_mean']
                }
                overall_data.append(row)
        
        df_overall = pd.DataFrame(overall_data)
        
        # Save overall comparison / 保存总体比较
        filepath = os.path.join(self.output_dir, 'model_comparisons', 'overall_comparison.csv')
        df_overall.to_csv(filepath, index=False, encoding='utf-8')
        
        # Create heatmap visualization / 创建热图可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Overall Model Performance Heatmap\n总体模型性能热图', fontsize=16, fontweight='bold')
        
        metrics = ['Pearson_R', 'R2_Score', 'RMSE', 'MAE']
        metric_labels = ['Pearson Correlation', 'R² Score', 'RMSE', 'MAE']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx//2, idx%2]
            
            # Create pivot table for heatmap / 为热图创建数据透视表
            pivot_data = df_overall.pivot(index='Target', columns='Model', values=metric)
            
            # Create heatmap / 创建热图
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
            ax.set_title(f'{label} Heatmap')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'model_comparisons', 'overall_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Model comparisons complete!")


def main():
    """
    Main function to run the comprehensive nanofiltration analysis
    运行综合纳滤分析的主函数
    """
    print("🌟 Starting Comprehensive Nanofiltration Membrane ML Analysis")
    print("🌟 开始综合纳滤膜机器学习分析")
    
    # Initialize analyzer / 初始化分析器
    analyzer = NanofiltrationMLAnalyzer(data_file='Data.xlsx')
    
    # Run comprehensive analysis / 运行综合分析
    analyzer.run_comprehensive_analysis()
    
    print("\n✅ Analysis completed successfully!")
    print("✅ 分析成功完成！")
    print(f"📁 Check results at: {analyzer.output_dir}")
    print(f"📁 请查看结果：{analyzer.output_dir}")


if __name__ == "__main__":
    main()