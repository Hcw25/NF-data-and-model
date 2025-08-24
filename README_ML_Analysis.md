# Comprehensive Nanofiltration Membrane ML Analysis System
# 综合纳滤膜机器学习分析系统

## Overview / 概述

This repository contains a comprehensive machine learning analysis system for predicting nanofiltration membrane properties, specifically designed to analyze membrane permeance and lithium-magnesium separation coefficient using multiple ML approaches.

本仓库包含一个用于预测纳滤膜性能的综合机器学习分析系统，专门用于使用多种ML方法分析膜渗透性和锂镁分离系数。

## Features / 功能特点

### 🎯 Target Variables / 目标变量
- **Permeance (渗透性)**: Membrane permeability characteristics
- **Lithium-Magnesium Separation Coefficient (锂镁分离系数)**: Ion selectivity performance

### 🤖 Machine Learning Models / 机器学习模型
1. **XGBoost Regression**: Gradient boosting with optimized hyperparameters
2. **Random Forest**: Ensemble method with feature importance analysis
3. **Neural Network (MLP)**: Multi-layer perceptron for non-linear modeling

### 📊 Analysis Features / 分析功能
- **Cross-Validation**: 300 experiments per model for robust evaluation
- **Performance Metrics**: Pearson correlation, R², RMSE, MAE
- **Feature Importance**: Detailed analysis of input variable contributions
- **SHAP Analysis**: Model interpretability and feature interaction insights
- **Statistical Analysis**: Mean and standard deviation across multiple runs
- **Professional Visualizations**: High-resolution bilingual charts and graphs

## Installation / 安装

### Prerequisites / 前提条件
```bash
pip install pandas openpyxl numpy scikit-learn xgboost matplotlib seaborn shap
```

### Required Python Packages / 所需Python包
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- shap >= 0.40.0
- openpyxl >= 3.0.0

## Usage / 使用方法

### Quick Start / 快速开始

1. **Run Complete Analysis / 运行完整分析**:
   ```bash
   python comprehensive_nf_analysis.py
   ```

2. **Test System / 测试系统**:
   ```bash
   python test_system.py
   ```

### Input Data / 输入数据

The system expects an Excel file `Data.xlsx` with the following structure:
系统需要一个Excel文件`Data.xlsx`，结构如下：

- **Target Columns / 目标列**:
  - `Permeance`: Membrane permeance values
  - `Lithium - Magnesium Separation Coefficient`: Separation performance values

- **Feature Columns / 特征列**:
  - Membrane properties (pore size, thickness, surface roughness, etc.)
  - Operating conditions (concentration, pressure, temperature, etc.)
  - Material characteristics (zeta potential, contact angle, MWCO, etc.)

## Output Structure / 输出结构

The analysis generates organized results in `~/Desktop/NF_Analysis_Results/`:

```
NF_Analysis_Results/
├── csv_results/                    # Detailed numerical results / 详细数值结果
│   ├── *_performance_summary.csv   # Performance metrics / 性能指标
│   ├── *_detailed_results.csv      # All experiment results / 所有实验结果
│   └── *_feature_importance.csv    # Feature importance rankings / 特征重要性排名
├── figures/                        # Visualizations / 可视化图表
│   ├── *_performance_distribution.png    # Metric distributions / 指标分布
│   ├── *_feature_importance.png          # Importance bar charts / 重要性条形图
│   └── *_prediction_scatter.png          # Prediction vs actual / 预测与实际
├── shap_analysis/                  # Model interpretability / 模型可解释性
│   ├── *_shap_values.csv          # SHAP values data / SHAP值数据
│   ├── *_shap_feature_importance.png     # SHAP importance / SHAP重要性
│   └── *_shap_summary.png         # SHAP summary plots / SHAP汇总图
└── model_comparisons/              # Model comparisons / 模型比较
    ├── overall_comparison.csv      # Cross-model comparison / 跨模型比较
    ├── overall_heatmap.png        # Performance heatmap / 性能热图
    └── *_model_comparison.png     # Individual target comparisons / 单个目标比较
```

## Key Results Summary / 关键结果摘要

Based on the analysis of the nanofiltration membrane dataset:

### Performance by Target / 按目标的性能

**Permeance Prediction / 渗透性预测**:
- Best Model: Random Forest (R² = -0.42)
- Challenge: High variability in permeance data
- Key Features: Total Concentration, Membrane Permeation Flux, Pore Size

**Li-Mg Separation Prediction / 锂镁分离预测**:
- Best Model: Random Forest (R² = 0.57, Pearson r = 0.80)
- Strong predictive performance achieved
- Key Features: Mg Ion Rejection Rate, Pore Size, Li Ion Rejection Rate

### Model Performance Comparison / 模型性能比较

| Model | Target | Pearson r | R² | RMSE | MAE |
|-------|--------|-----------|----|----- |----- |
| Random Forest | Separation | 0.80 ± 0.09 | 0.57 ± 0.21 | 20.6 ± 4.8 | 14.4 ± 3.2 |
| XGBoost | Separation | 0.78 ± 0.09 | 0.51 ± 0.25 | 21.8 ± 5.2 | 14.8 ± 3.4 |
| Random Forest | Permeance | 0.11 ± 0.20 | -0.42 ± 0.72 | 8.6 ± 3.0 | 6.0 ± 1.5 |

## Code Structure / 代码结构

### Main Classes / 主要类

**`NanofiltrationMLAnalyzer`**: Core analysis engine
- Data preprocessing and cleaning / 数据预处理和清理
- Model training with cross-validation / 带交叉验证的模型训练
- Performance evaluation and statistics / 性能评估和统计
- Visualization generation / 可视化生成
- Results export and organization / 结果导出和整理

### Key Methods / 关键方法

- `load_and_preprocess_data()`: Data cleaning and preparation
- `train_and_evaluate_model()`: ML model training and evaluation
- `generate_shap_analysis()`: Model interpretability analysis
- `generate_model_comparisons()`: Comparative analysis across models

## Validation and Testing / 验证和测试

The system includes comprehensive testing:
- Data integrity validation / 数据完整性验证
- Model performance verification / 模型性能验证
- Output file generation testing / 输出文件生成测试
- Cross-platform compatibility / 跨平台兼容性

## Scientific Insights / 科学见解

### Feature Importance Findings / 特征重要性发现

**For Permeance / 渗透性方面**:
1. Total Concentration (38%) - Dominant factor
2. Membrane Permeation Flux (19%) - Secondary importance
3. Pore Size (8%) - Structural influence

**For Li-Mg Separation / 锂镁分离方面**:
1. Magnesium Ion Rejection Rate (37%) - Primary predictor
2. Pore Size (18%) - Critical structural parameter
3. Lithium Ion Rejection Rate (16%) - Complementary factor

### Model Insights / 模型见解

- **Random Forest** consistently outperforms other models due to:
  - Better handling of feature interactions / 更好地处理特征交互
  - Robustness to outliers / 对异常值的鲁棒性
  - Ensemble averaging reduces overfitting / 集成平均减少过拟合

- **Separation coefficient** is more predictable than permeance, suggesting:
  - Clearer relationship with membrane properties / 与膜性质的关系更清晰
  - Less experimental noise in separation measurements / 分离测量中的实验噪声较少

## Future Enhancements / 未来改进

Potential improvements for the system:
- Deep learning models (CNN, LSTM) for sequential data
- Bayesian optimization for hyperparameter tuning  
- Multi-objective optimization for membrane design
- Integration with experimental design workflows
- Real-time prediction capabilities

## Authors and Acknowledgments / 作者和致谢

Developed as part of nanofiltration membrane research initiative.
Based on experimental data and methodologies from membrane science research.

## License / 许可证

This project is provided for research and educational purposes.

## Contact / 联系方式

For questions, improvements, or collaboration opportunities, please refer to the repository issues section.

---

**Note**: This system is designed for research purposes and should be validated against additional experimental data for production applications.

**注意**: 本系统设计用于研究目的，在生产应用中应针对额外的实验数据进行验证。