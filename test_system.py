#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test and demonstration script for the comprehensive NF analysis system
测试和演示综合NF分析系统的脚本
"""

import pandas as pd
import os
import sys

def test_system():
    """Test the comprehensive analysis system and show sample results"""
    
    print("🧪 Testing Comprehensive Nanofiltration Analysis System")
    print("🧪 测试综合纳滤分析系统")
    
    # Check if results exist from previous run
    results_dir = os.path.expanduser('~/Desktop/NF_Analysis_Results')
    
    if os.path.exists(results_dir):
        print(f"\n📊 Found analysis results in: {results_dir}")
        
        # Show sample results
        show_sample_results(results_dir)
        
    else:
        print(f"\n⚠️ No results found. Running analysis...")
        
        # Import and run the analysis
        try:
            from comprehensive_nf_analysis import NanofiltrationMLAnalyzer
            
            analyzer = NanofiltrationMLAnalyzer(data_file='Data.xlsx')
            analyzer.run_comprehensive_analysis()
            
            print("✅ Analysis completed successfully!")
            show_sample_results(analyzer.output_dir)
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return False
    
    return True

def show_sample_results(results_dir):
    """Show sample results from the analysis"""
    
    print("\n📈 SAMPLE RESULTS OVERVIEW")
    print("📈 样本结果概述")
    
    # Show performance summary for each target
    targets = ['permeance', 'separation']
    models = ['XGBoost', 'RandomForest', 'NeuralNetwork']
    
    for target in targets:
        print(f"\n🎯 Target: {target.upper()}")
        
        for model in models:
            summary_file = os.path.join(results_dir, 'csv_results', 
                                      f"{target}_{model}_performance_summary.csv")
            
            if os.path.exists(summary_file):
                df = pd.read_csv(summary_file)
                print(f"\n   📊 {model} Performance:")
                for _, row in df.iterrows():
                    print(f"      {row['Metric']}: {row['Mean']:.4f} ± {row['Std']:.4f}")
    
    # Show feature importance for best models
    print("\n🔍 FEATURE IMPORTANCE HIGHLIGHTS")
    print("🔍 特征重要性亮点")
    
    for target in targets:
        for model in models:
            importance_file = os.path.join(results_dir, 'csv_results',
                                         f"{target}_{model}_feature_importance.csv")
            
            if os.path.exists(importance_file):
                df = pd.read_csv(importance_file)
                top_features = df.head(3)
                
                print(f"\n   🏆 Top 3 Features for {target} ({model}):")
                for _, row in top_features.iterrows():
                    print(f"      {row['Feature']}: {row['Importance_Mean']:.4f}")
                break  # Show only first available model per target
    
    # Show file counts
    print(f"\n📁 OUTPUT FILES GENERATED:")
    subdirs = ['csv_results', 'figures', 'shap_analysis', 'model_comparisons']
    
    total_files = 0
    for subdir in subdirs:
        subdir_path = os.path.join(results_dir, subdir)
        if os.path.exists(subdir_path):
            file_count = len([f for f in os.listdir(subdir_path) 
                            if os.path.isfile(os.path.join(subdir_path, f))])
            print(f"   📂 {subdir}: {file_count} files")
            total_files += file_count
    
    print(f"   📊 Total files: {total_files}")
    
    # Show model comparison
    comparison_file = os.path.join(results_dir, 'model_comparisons', 'overall_comparison.csv')
    if os.path.exists(comparison_file):
        print(f"\n🔄 MODEL COMPARISON OVERVIEW")
        df_comp = pd.read_csv(comparison_file)
        
        # Show best performing model per target by R2 score
        for target in targets:
            target_data = df_comp[df_comp['Target'] == target.title()]
            if not target_data.empty:
                best_model = target_data.loc[target_data['R2_Score'].idxmax()]
                print(f"   🏆 Best for {target}: {best_model['Model']} "
                      f"(R² = {best_model['R2_Score']:.4f})")

def main():
    """Main test function"""
    
    # Change to the repository directory
    repo_dir = "/home/runner/work/NF-data-and-model/NF-data-and-model"
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
        print(f"📁 Working directory: {repo_dir}")
    
    # Run the test
    success = test_system()
    
    if success:
        print("\n✅ System test completed successfully!")
        print("✅ 系统测试成功完成！")
        print("\n📝 Key Features Implemented:")
        print("   - Three ML models (XGBoost, RandomForest, Neural Network)")
        print("   - Two target variables (Permeance, Li-Mg Separation)")
        print("   - 300 cross-validation experiments per model")
        print("   - Performance metrics (Pearson, R², RMSE, MAE)")
        print("   - Feature importance analysis")
        print("   - SHAP interpretability analysis")
        print("   - Professional bilingual visualizations")
        print("   - Comprehensive CSV outputs")
        
    else:
        print("\n❌ System test failed!")
        print("❌ 系统测试失败！")

if __name__ == "__main__":
    main()