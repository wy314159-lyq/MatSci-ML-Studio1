"""
Intelligent Wizard System for MatSci-ML Studio
智能化向导系统，提供数据分析和配置推荐
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QLabel, QPushButton, QTextEdit, QProgressBar,
                            QListWidget, QListWidgetItem, QCheckBox, QComboBox,
                            QSpinBox, QDoubleSpinBox, QSlider, QTabWidget,
                            QScrollArea, QFrame, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont

class DataAnalysisWorker(QThread):
    """智能数据分析工作线程"""
    analysis_completed = pyqtSignal(dict)
    recommendation_ready = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        
    def run(self):
        """执行智能数据分析"""
        try:
            self.status_updated.emit("🔍 Analyzing data features...")
            self.progress_updated.emit(10)
            
            # 1. Basic data statistics
            data_stats = self._analyze_data_statistics()
            self.progress_updated.emit(25)
            
            # 2. Data quality assessment
            self.status_updated.emit("📊 Assessing data quality...")
            quality_assessment = self._assess_data_quality()
            self.progress_updated.emit(50)
            
            # 3. Task type inference
            self.status_updated.emit("🎯 Inferring task type...")
            task_type = self._infer_task_type()
            self.progress_updated.emit(70)
            
            # 4. Generate intelligent recommendations
            self.status_updated.emit("🧠 Generating intelligent recommendations...")
            recommendations = self._generate_recommendations(data_stats, quality_assessment, task_type)
            self.progress_updated.emit(100)
            
            # 汇总分析结果
            analysis_result = {
                'data_stats': data_stats,
                'quality_assessment': quality_assessment,
                'task_type': task_type,
                'recommendations': recommendations
            }
            
            self.analysis_completed.emit(analysis_result)
            self.recommendation_ready.emit(recommendations)
            
        except Exception as e:
            self.status_updated.emit(f"❌ Analysis failed: {str(e)}")
    
    def _analyze_data_statistics(self):
        """分析数据统计信息"""
        stats = {
            'n_samples': len(self.X),
            'n_features': len(self.X.columns),
            'feature_types': {},
            'target_type': self._get_target_type(),
            'memory_usage': self.X.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # 分析特征类型
        for col in self.X.columns:
            dtype = self.X[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                if self.X[col].nunique() <= 10:
                    stats['feature_types'][col] = 'categorical_numeric'
                else:
                    stats['feature_types'][col] = 'continuous'
            else:
                stats['feature_types'][col] = 'categorical'
        
        return stats
    
    def _assess_data_quality(self):
        """评估数据质量"""
        assessment = {
            'missing_data': {},
            'outliers': {},
            'duplicates': self.X.duplicated().sum(),
            'correlation_issues': [],
            'overall_score': 0,
            'issues': [],
            'recommendations': []
        }
        
        # 缺失数据分析
        missing_info = self.X.isnull().sum()
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                missing_pct = (missing_count / len(self.X)) * 100
                assessment['missing_data'][col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
                if missing_pct > 50:
                    assessment['issues'].append(f"Feature {col} has high missing rate ({missing_pct:.1f}%)")
                    assessment['recommendations'].append(f"Consider removing feature {col} or using advanced imputation methods")
        
        # 异常值检测 (简单的IQR方法)
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.X[col].quantile(0.25)
            Q3 = self.X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.X[col] < lower_bound) | (self.X[col] > upper_bound)).sum()
            
            if outliers > 0:
                outlier_pct = (outliers / len(self.X)) * 100
                assessment['outliers'][col] = {
                    'count': outliers,
                    'percentage': outlier_pct
                }
                if outlier_pct > 10:
                    assessment['issues'].append(f"Feature {col} has many outliers ({outlier_pct:.1f}%)")
                    assessment['recommendations'].append(f"Check for outliers in feature {col} and consider using robust preprocessing methods")
        
        # 相关性问题检测
        if len(numeric_cols) > 1:
            corr_matrix = self.X[numeric_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                assessment['correlation_issues'] = high_corr_pairs
                assessment['issues'].append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
                assessment['recommendations'].append("Enable correlation filtering to remove redundant features")
        
        # 计算总体质量分数
        base_score = 100
        base_score -= min(30, len(assessment['issues']) * 10)  # Each issue deducts 10 points, up to 30 points
        base_score -= min(20, assessment['duplicates'] / len(self.X) * 100)  # Duplicate data deduction
        assessment['overall_score'] = max(0, base_score)
        
        return assessment
    
    def _infer_task_type(self):
        """推断任务类型"""
        if self.y is None:
            return 'unsupervised'
        
        # 检查目标变量类型
        if pd.api.types.is_numeric_dtype(self.y.dtype):
            unique_values = self.y.nunique()
            if unique_values <= 10:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'
    
    def _get_target_type(self):
        """获取目标变量类型"""
        if self.y is None:
            return 'none'
        
        if pd.api.types.is_numeric_dtype(self.y.dtype):
            return 'numeric'
        else:
            return 'categorical'
    
    def _generate_recommendations(self, data_stats, quality_assessment, task_type):
        """生成智能推荐"""
        recommendations = {
            'preprocessing': [],
            'feature_selection': [],
            'models': [],
            'hyperparameters': {},
            'workflow': [],
            'performance_tips': []
        }
        
        n_samples = data_stats['n_samples']
        n_features = data_stats['n_features']
        
        # 预处理建议
        if quality_assessment['missing_data']:
            recommendations['preprocessing'].append({
                'step': 'missing_data_handling',
                'description': 'Handle missing data',
                'method': 'median_imputation' if len(quality_assessment['missing_data']) < n_features * 0.3 else 'advanced_imputation',
                'priority': 'high'
            })
        
        if quality_assessment['outliers']:
            recommendations['preprocessing'].append({
                'step': 'outlier_treatment',
                'description': 'Handle outliers',
                'method': 'robust_scaling',
                'priority': 'medium'
            })
        
        # 数据缩放建议
        numeric_features = sum(1 for ftype in data_stats['feature_types'].values() if ftype == 'continuous')
        if numeric_features > 0:
            recommendations['preprocessing'].append({
                'step': 'scaling',
                'description': 'Feature scaling',
                'method': 'StandardScaler' if n_samples > 100 else 'RobustScaler',
                'priority': 'high'
            })
        
        # 特征选择建议
        if n_features > 50:
            recommendations['feature_selection'].append({
                'method': 'importance_filtering',
                'description': 'Feature filtering based on importance',
                'params': {'top_k': min(30, n_features // 2)},
                'priority': 'high'
            })
        
        if quality_assessment['correlation_issues']:
            recommendations['feature_selection'].append({
                'method': 'correlation_filtering',
                'description': 'Correlation filtering',
                'params': {'threshold': 0.95},
                'priority': 'high'
            })
        
        if n_features <= 15 and n_samples < 10000:
            recommendations['feature_selection'].append({
                'method': 'exhaustive_search',
                'description': 'Exhaustive search for optimal feature combinations',
                'params': {'min_features': 3, 'max_features': min(10, n_features)},
                'priority': 'medium'
            })
        elif n_features <= 50:
            recommendations['feature_selection'].append({
                'method': 'genetic_algorithm',
                'description': 'Genetic algorithm feature selection',
                'params': {'population_size': 30, 'generations': 15},
                'priority': 'medium'
            })
        
        # 模型推荐
        if task_type == 'classification':
            if n_samples < 1000:
                recommendations['models'] = ['Logistic Regression', 'Random Forest', 'Support Vector Machine']
            else:
                recommendations['models'] = ['Random Forest', 'XGBoost', 'LightGBM']
        elif task_type == 'regression':
            if n_samples < 1000:
                recommendations['models'] = ['Linear Regression', 'Random Forest', 'Support Vector Regression']
            else:
                recommendations['models'] = ['Random Forest', 'XGBoost', 'LightGBM']
        
        # 超参数建议
        recommendations['hyperparameters'] = {
            'cv_folds': 5 if n_samples > 500 else 3,
            'search_method': 'Random Search' if n_samples > 100 else 'Grid Search',
            'n_iter': 50 if n_samples > 1000 else 30
        }
        
        # 工作流建议
        workflow_steps = [
            "1. Data quality check and cleaning",
            "2. Feature preprocessing and scaling", 
            "3. Feature selection (multi-stage strategy)",
            "4. Model training and validation",
            "5. Hyperparameter optimization",
            "6. Model evaluation and interpretation"
        ]
        recommendations['workflow'] = workflow_steps
        
        # 性能优化建议
        if n_samples > 1000 or n_features > 20:
            recommendations['performance_tips'].append("Enable parallel processing to speed up calculations")
        
        if n_samples > 5000:
            recommendations['performance_tips'].append("Consider reducing the number of cross-validation folds to speed up the process")
        
        if n_features > 100:
            recommendations['performance_tips'].append("Use feature selection to reduce dimensionality")
        
        return recommendations

class IntelligentWizard(QWidget):
    """智能向导主界面"""
    
    # 信号
    configuration_ready = pyqtSignal(dict)
    wizard_completed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.X = None
        self.y = None
        self.analysis_results = None
        self.recommendations = None
        self.worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("🧙‍♂️ Intelligent Wizard System")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { color: #2196F3; margin: 10px; }")
        layout.addWidget(title)
        
        # 创建标签页
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 1. 数据分析页
        self.create_analysis_tab()
        
        # 2. 推荐配置页
        self.create_recommendations_tab()
        
        # 3. 工作流页
        self.create_workflow_tab()
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #666666; margin: 5px; }")
        layout.addWidget(self.status_label)
        
        # 控制按钮
        self.create_control_buttons(layout)
    
    def create_analysis_tab(self):
        """创建数据分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Data overview
        overview_group = QGroupBox("📊 Data Overview")
        overview_layout = QVBoxLayout(overview_group)
        
        self.data_overview_text = QTextEdit()
        self.data_overview_text.setMaximumHeight(150)
        self.data_overview_text.setReadOnly(True)
        overview_layout.addWidget(self.data_overview_text)
        
        layout.addWidget(overview_group)
        
        # Data quality assessment
        quality_group = QGroupBox("🔍 Data Quality Assessment")
        quality_layout = QVBoxLayout(quality_group)
        
        self.quality_assessment_text = QTextEdit()
        self.quality_assessment_text.setReadOnly(True)
        quality_layout.addWidget(self.quality_assessment_text)
        
        layout.addWidget(quality_group)
        
        # Issues and suggestions
        issues_group = QGroupBox("⚠️ Issues and Suggestions")
        issues_layout = QVBoxLayout(issues_group)
        
        self.issues_list = QListWidget()
        issues_layout.addWidget(self.issues_list)
        
        layout.addWidget(issues_group)
        
        self.tabs.addTab(tab, "Data Analysis")
    
    def create_recommendations_tab(self):
        """创建推荐配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 智能推荐
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Preprocessing recommendations
        self.preprocessing_group = QGroupBox("🔧 Preprocessing Recommendations")
        preprocessing_layout = QVBoxLayout(self.preprocessing_group)
        self.preprocessing_checks = {}
        scroll_layout.addWidget(self.preprocessing_group)
        
        # Feature selection recommendations
        self.feature_selection_group = QGroupBox("🎯 Feature Selection Recommendations")
        fs_layout = QVBoxLayout(self.feature_selection_group)
        self.feature_selection_checks = {}
        scroll_layout.addWidget(self.feature_selection_group)
        
        # Model recommendations
        self.model_group = QGroupBox("🤖 Model Recommendations")
        model_layout = QVBoxLayout(self.model_group)
        self.model_checks = {}
        scroll_layout.addWidget(self.model_group)
        
        # Hyperparameter recommendations
        self.hyperparams_group = QGroupBox("⚙️ Hyperparameter Recommendations")
        hyperparams_layout = QVBoxLayout(self.hyperparams_group)
        self.hyperparams_widgets = {}
        scroll_layout.addWidget(self.hyperparams_group)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        self.tabs.addTab(tab, "Intelligent Recommendations")
    
    def create_workflow_tab(self):
        """创建工作流标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Recommended workflow
        workflow_group = QGroupBox("📋 Recommended Workflow")
        workflow_layout = QVBoxLayout(workflow_group)
        
        self.workflow_list = QListWidget()
        workflow_layout.addWidget(self.workflow_list)
        
        layout.addWidget(workflow_group)
        
        # Performance optimization tips
        tips_group = QGroupBox("⚡ Performance Optimization Tips")
        tips_layout = QVBoxLayout(tips_group)
        
        self.performance_tips_list = QListWidget()
        tips_layout.addWidget(self.performance_tips_list)
        
        layout.addWidget(tips_group)
        
        self.tabs.addTab(tab, "Workflow Guide")
    
    def create_control_buttons(self, parent_layout):
        """创建控制按钮"""
        button_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("🔍 Start Intelligent Analysis")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        button_layout.addWidget(self.analyze_btn)
        
        self.apply_btn = QPushButton("✅ Apply Recommended Configuration")
        self.apply_btn.clicked.connect(self.apply_recommendations)
        self.apply_btn.setEnabled(False)
        button_layout.addWidget(self.apply_btn)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_wizard)
        button_layout.addWidget(self.reset_btn)
        
        parent_layout.addLayout(button_layout)
    
    def set_data(self, X: pd.DataFrame, y: pd.Series = None):
        """设置数据"""
        self.X = X
        self.y = y
        self.analyze_btn.setEnabled(True)
        
        # 显示基本数据信息
        basic_info = f"""Data dimension: {X.shape[0]} samples × {X.shape[1]} features
Target variable: {'Yes' if y is not None else 'No'}
Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB"""
        
        self.data_overview_text.setText(basic_info)
        self.status_label.setText("Data loaded, click Start Analysis")
    
    def start_analysis(self):
        """开始智能分析"""
        if self.X is None:
            return
        
        self.progress_bar.setVisible(True)
        self.analyze_btn.setEnabled(False)
        
        # 创建分析工作线程
        self.worker = DataAnalysisWorker(self.X, self.y)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.analysis_completed.connect(self.on_analysis_completed)
        self.worker.recommendation_ready.connect(self.on_recommendations_ready)
        self.worker.start()
    
    def on_analysis_completed(self, results):
        """分析完成回调"""
        self.analysis_results = results
        
        # 更新数据质量评估
        quality = results['quality_assessment']
        quality_text = f"""Overall quality score: {quality['overall_score']}/100

Missing data: {len(quality['missing_data'])} features have missing values
Outliers: {len(quality['outliers'])} features have outliers
Duplicates: {quality['duplicates']}

Task type: {results['task_type']}"""
        
        self.quality_assessment_text.setText(quality_text)
        
        # 更新问题列表
        self.issues_list.clear()
        for issue in quality['issues']:
            item = QListWidgetItem(f"⚠️ {issue}")
            self.issues_list.addItem(item)
        
        for rec in quality['recommendations']:
            item = QListWidgetItem(f"💡 {rec}")
            self.issues_list.addItem(item)
        
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Analysis completed! View recommended configuration")
    
    def on_recommendations_ready(self, recommendations):
        """推荐准备就绪回调"""
        self.recommendations = recommendations
        self.update_recommendations_ui()
        self.apply_btn.setEnabled(True)
    
    def update_recommendations_ui(self):
        """更新推荐界面"""
        if not self.recommendations:
            return
        
        # 清空现有组件
        self.clear_recommendation_widgets()
        
        # 预处理推荐
        for rec in self.recommendations['preprocessing']:
            cb = QCheckBox(f"{rec['description']} ({rec['method']})")
            cb.setChecked(rec['priority'] == 'high')
            cb.setToolTip(f"Priority: {rec['priority']}")
            self.preprocessing_checks[rec['step']] = cb
            self.preprocessing_group.layout().addWidget(cb)
        
        # 特征选择推荐
        for rec in self.recommendations['feature_selection']:
            cb = QCheckBox(f"{rec['description']}")
            cb.setChecked(rec['priority'] == 'high')
            cb.setToolTip(f"Priority: {rec['priority']}")
            self.feature_selection_checks[rec['method']] = cb
            self.feature_selection_group.layout().addWidget(cb)
        
        # 模型推荐
        for model in self.recommendations['models']:
            cb = QCheckBox(model)
            cb.setChecked(True)
            self.model_checks[model] = cb
            self.model_group.layout().addWidget(cb)
        
        # 超参数推荐
        hyperparams = self.recommendations['hyperparameters']
        for param, value in hyperparams.items():
            label = QLabel(f"{param}: {value}")
            self.hyperparams_group.layout().addWidget(label)
        
        # 工作流
        self.workflow_list.clear()
        for step in self.recommendations['workflow']:
            item = QListWidgetItem(step)
            self.workflow_list.addItem(item)
        
        # 性能提示
        self.performance_tips_list.clear()
        for tip in self.recommendations['performance_tips']:
            item = QListWidgetItem(f"⚡ {tip}")
            self.performance_tips_list.addItem(item)
    
    def clear_recommendation_widgets(self):
        """清空推荐组件"""
        # 清空预处理组件
        for cb in self.preprocessing_checks.values():
            cb.deleteLater()
        self.preprocessing_checks.clear()
        
        # 清空特征选择组件  
        for cb in self.feature_selection_checks.values():
            cb.deleteLater()
        self.feature_selection_checks.clear()
        
        # 清空模型组件
        for cb in self.model_checks.values():
            cb.deleteLater()
        self.model_checks.clear()
        
        # 清空超参数组件
        for widget in self.hyperparams_widgets.values():
            widget.deleteLater()
        self.hyperparams_widgets.clear()
    
    def apply_recommendations(self):
        """应用推荐配置"""
        if not self.recommendations:
            return
        
        config = self.generate_configuration()
        self.configuration_ready.emit(config)
        self.wizard_completed.emit()
    
    def generate_configuration(self):
        """生成配置"""
        config = {
            'task_type': self.analysis_results['task_type'],
            'selected_models': [model for model, cb in self.model_checks.items() if cb.isChecked()],
            'preprocessing': {
                step: cb.isChecked() 
                for step, cb in self.preprocessing_checks.items()
            },
            'feature_selection': {
                method: cb.isChecked() 
                for method, cb in self.feature_selection_checks.items()
            },
            'hyperparameters': self.recommendations['hyperparameters'],
            'analysis_results': self.analysis_results
        }
        
        return config
    
    def reset_wizard(self):
        """重置向导"""
        self.X = None
        self.y = None
        self.analysis_results = None
        self.recommendations = None
        
        self.data_overview_text.clear()
        self.quality_assessment_text.clear()
        self.issues_list.clear()
        self.workflow_list.clear()
        self.performance_tips_list.clear()
        
        self.clear_recommendation_widgets()
        
        self.analyze_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.status_label.setText("Ready")
        self.progress_bar.setVisible(False) 