# AutoMatFlow (MatSci-ML Studio) - v1
面向材料科学的图形化机器学习工作流平台（GUI），入口为 `main.py`（会启动 `ui/main_window.py` 的主窗口）。

**运行方式**: `python main.py`
**建议 Python 版本**: 3.10–3.12（优先 3.12；3.13 取决于 PyQt5/依赖轮子支持情况）
**依赖说明**: 本仓库当前未提供 `requirements.txt`；安装方式见下方“快速开始”。

## 🚀 快速开始（推荐）

```bash
# conda（推荐）
conda create -n automl python=3.10 -c conda-forge
conda activate automl

# 或 venv（Windows PowerShell）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 核心依赖（GUI + 数据处理 + 传统 ML）
pip install -U pip
pip install PyQt5 numpy pandas scikit-learn matplotlib seaborn joblib scipy openpyxl xlrd

# 可选功能（按需安装）
pip install shap psutil xgboost lightgbm catboost pymoo scikit-optimize umap-learn hdbscan

# 可选深度学习模块
pip install torch pymatgen  # CGCNN
pip install torch jarvis-tools pydantic pydantic-settings tqdm  # ALIGNN（DGL 安装参考官方 wheel）

# 启动
python main.py
```

> 注：README 下方仍保留了一段历史版本的“快速开始”内容，可能包含过时信息；以本节为准。

## ⚠️ 历史内容提示

由于 README 合并过多次版本，文档下半部分仍可能出现以下**已过时**的表述（以本页顶部的“快速开始（推荐）/架构概览/模块列表”为准）：

- **Python 3.13+**：当前更建议使用 **Python 3.10**，以获得更好的依赖兼容性。

- **主动学习 GUI**（如 `active_learning_window.py` / `ActiveLearningWindow`）：当前仓库提供的是**非 GUI** 的主动学习核心算法（`modules/active_learning.py`），尚未接入主界面标签页。
- **模块清单**：当前主界面额外包含“智能聚类分析”、以及可选的 “ALIGNN/CGCNN” 标签页（依赖缺失时会降级提示）。

## 🧱 架构概览

- **入口**: `main.py` 负责设置路径并调用 `ui/main_window.py:main()` 启动 GUI。
- **主窗口**: `ui/main_window.py` 负责菜单/标签页初始化，并通过 Qt 信号/槽连接各模块数据流。
- **业务模块**: `modules/` 下每个模块通常对应一个 `QWidget` 标签页；耗时任务放在 `QThread` 中执行，避免阻塞 UI。
- **通用工具**: `utils/` 提供数据导入与绘图、模型/预测辅助、特征名映射等跨模块能力。

**模块间数据流（简化）**:

- `DataModule.data_ready` → `IntelligentWizard.set_data` / `AdvancedPreprocessing.set_data` / `FeatureModule.set_data` /（原始数据）`IntelligentClusteringModule`
- `AdvancedPreprocessing.preprocessing_completed` → `FeatureModule.set_data`
- `FeatureModule.features_ready` → `TrainingModule.set_data` / `SHAPAnalysisModule.set_data`
- `TrainingModule.model_ready` → `PredictionModule.set_model` / `SHAPAnalysisModule.set_model` / `TargetOptimizationModule.set_model`

**可选模块降级策略**:

- `modules/alignn` 与 `modules/cgcnn` 在依赖缺失或 DLL 加载失败时，会在主界面显示“Unavailable”占位页，并给出安装提示。

## ✨ 项目特色

*   **模块化工作流**: 清晰地划分了数据管理、特征工程、模型训练、预测和优化等多个阶段，符合科研的逻辑流程。
*   **图形化界面 (GUI)**: 所有操作均可通过直观的图形界面完成，无需编写复杂的代码。
*   **智能化向导**: 内置智能分析与推荐系统，可自动评估数据质量并推荐合适的预处理、特征选择和模型配置。
*   **高级优化算法**: 集成了遗传算法、粒子群优化、贝叶斯优化以及多目标优化（NSGA-II, MOEA/D 等），用于目标性能的反向设计和优化。
*   **模型可解释性**: 提供了基于 SHAP (SHapley Additive exPlanations) 的模型解释功能，帮助用户理解模型的决策过程。
*   **全面的预处理**: 支持高级异常值检测、智能缺失值填充、数据变换和特征缩放。
*   **协作与版本控制**: 内置项目管理和版本快照功能，便于团队协作和实验追溯。
*   **实时性能监控**: 实时监控系统资源占用和任务进度，确保流畅的运行体验。

---

## 📚 模块详解

本项目主要由以下模块构成（其中部分为可选功能，缺少依赖时会在界面中提示不可用）：

- 📊 数据管理（Data Management）
- 💡 智能向导（Intelligent Wizard）
- 🧬 高级预处理（Advanced Preprocessing）
- 🎯 特征工程与选择（Feature Selection）
- 🔬 模型训练与评估（Model Training）
- ✅ 模型预测与结果导出（Prediction）
- 🧠 SHAP 模型可解释性分析（SHAP）
- 🎯 目标属性优化（Target Optimization）
- 🔄 多目标优化（Multi-Objective Optimization）
- 🧩 智能聚类分析（Clustering Analysis）
- ⚙️ 性能监控（Performance Monitor）
- 🤝 协作与版本控制（Collaboration & Version Control）
- 🧱 ALIGNN 图神经网络（可选，需 `torch`/`dgl`/`jarvis-tools` 等依赖；见 `modules/alignn/README.md`）
- 💎 CGCNN 图神经网络（可选，需 `torch`/`pymatgen` 等依赖）
- 🤖 主动学习核心算法（实验性，当前未接入主界面标签页；见 `modules/active_learning.py`）

---

### 📊 模块一：数据管理与预处理

这是整个工作流的起点，负责数据的导入、探索、清洗和初步准备。

*   **核心功能**:
    *   **数据导入**: 支持从 CSV、Excel 文件以及系统剪贴板导入数据。
    *   **数据探索**: 提供数据概览（形状、类型、缺失值）和表格预览。
    *   **数据可视化**: 自动生成数据质量报告，包括缺失值热图、分布图、相关性矩阵等，帮助用户直观理解数据。
    *   **数据清洗**: 提供缺失值处理（删除、均值/中位数/众数填充）和重复行移除功能。
    *   **特征/目标选择**: 用户可以通过界面选择用于建模的特征列（X）和目标列（y）。
    *   **类型推荐**: 根据目标列的特性，自动推荐任务类型（分类或回归）。
    *   **分类特征编码**: 支持对分类特征进行独热编码（One-Hot）、标签编码（Label Encoding）等。

*   **核心类**: `DataModule(QWidget)`

---

### 🧬 模块二：高级数据预处理

本模块提供了一系列高级的数据清洗和变换工具，用于提升数据质量。

*   **核心功能**:
    *   **智能推荐系统**: 内置 `SmartRecommendationEngine`，可自动分析数据质量并推荐合适的预处理步骤。
    *   **高级异常值检测**: 集成了多种异常值检测算法，如 `IQR`, `Z-Score`, `Isolation Forest`, `Local Outlier Factor` 等，并提供可视化。
    *   **智能缺失值填充**: 除了基础填充方法，还支持 `KNNImputer` 和 `IterativeImputer` 等高级填充策略。
    *   **数据变换**: 提供 `StandardScaler`, `MinMaxScaler`, `PowerTransformer` 等多种数据缩放和变换工具，以满足不同模型的需求。
    *   **状态管理与历史记录**: `StateManager` 和 `OperationHistory` 实现了强大的撤销/重做功能和详细的操作日志，确保所有预处理步骤都可追溯。

*   **核心类**: `AdvancedPreprocessing(QWidget)`, `OutlierDetector`, `SmartImputer`, `StateManager`

---

### 🎯 模块三：特征工程与选择

此模块专注于从原始特征中筛选出对模型最有价值的子集，以提高模型性能并降低过拟合风险。

*   **核心功能**:
    *   **多阶段选择策略**: 实现了"重要性过滤" -> "相关性过滤" -> "高级搜索"的多阶段特征选择流程。
    *   **模型驱动的重要性评估**: 使用机器学习模型（如随机森林）来计算特征的重要性得分，并据此进行筛选。
    *   **相关性分析**: 自动检测并移除高度相关的冗余特征，提供过滤前后的相关性矩阵对比图。
    *   **高级子集搜索**:
        *   **序列特征选择 (SFS)**: 支持前向和后向的贪心搜索策略。
        *   **遗传算法 (GA)**: 通过模拟生物进化过程来搜索最优的特征组合。
        *   **穷举搜索**: 对特征子集进行暴力搜索（适用于特征数量较少的情况）。
    *   **并行计算**: 利用 `ThreadPoolExecutor` 和 `ProcessPoolExecutor` 加速计算密集型的特征评估过程。

*   **核心类**: `FeatureModule(QWidget)`, `FeatureSelectionWorker(QThread)`

---

### 🔬 模块四：模型训练与评估

这是机器学习流程的核心，负责使用处理好的数据来训练模型，并对其性能进行全面评估。

*   **核心功能**:
    *   **丰富的模型库**: 通过 `ml_utils` 提供了全面的分类和回归模型库，包括线性模型、SVM、树模型、集成模型（随机森林、梯度提升）、神经网络等。
    *   **自动化预处理管道**: 自动为数值和分类特征构建 `sklearn.pipeline.Pipeline`，集成了缺失值填充、缩放和编码。
    *   **超参数优化 (HPO)**: 支持网格搜索、随机搜索和贝叶斯搜索等多种方法，自动寻找模型的最佳参数。
    *   **可靠的性能评估**: 提供多种评估策略，包括单次划分、多次随机划分和嵌套交叉验证，以获得模型性能的无偏估计。
    *   **全面的评估指标与可视化**: 自动计算并展示各种指标（如 R², MSE, Accuracy, F1-Score, ROC AUC），并生成混淆矩阵、ROC曲线、残差图等多种可视化图表。
    *   **模型持久化**: 支持将训练好的完整管道（包含预处理器和模型）保存为 `.joblib` 文件，以便在预测模块中复用。

*   **核心类**: `TrainingModule(QWidget)`

---

### 🧠 模块五：SHAP 模型可解释性分析

此模块利用 SHAP (SHapley Additive exPlanations) 框架，为训练好的“黑箱”模型提供深入、直观的解释。

*   **核心功能**:
    *   **自动选择解释器**: 根据模型类型（树模型、线性模型等）自动选择最合适的 SHAP 解释器（`TreeExplainer`, `KernelExplainer` 等）。
    *   **全局解释**:
        *   **Summary Plot / Beeswarm Plot**: 直观展示每个特征对模型输出的总体影响方向和大小。
        *   **Feature Importance**: 以条形图形式清晰地展示特征的平均重要性排序。
    *   **局部解释**:
        *   **Waterfall Plot**: 详细分解单个样本的预测值，展示每个特征如何将其从基线值推向最终预测值。
        *   **Force Plot**: 直观地展示哪些特征是推动或拉低单个样本预测值的主要力量。
    *   **特征依赖图 (Dependence Plots)**: 揭示单个特征的边际效应对模型输出的影响，并能自动发现和可视化与其他特征的交互效应。
    *   **独立可视化窗口**: 解决 Matplotlib 与 PyQt 的兼容性问题，所有 SHAP 图都可以在独立的、可交互的窗口中打开和导出。

*   **核心类**: `SHAPAnalysisModule(QWidget)`, `SHAPCalculationWorker(QThread)`

---

### 🎯 模块六：目标属性优化

这是一个强大的反向设计工具，用于寻找能使材料达到特定目标属性的最佳特征组合。

*   **核心功能**:
    *   **多种优化算法**: 集成了遗传算法（GA）、粒子群优化（PSO），以及多种 `scipy` 和 `scikit-optimize` 中的高级优化器（如差分进化、贝叶斯优化）。
    *   **约束优化**: 支持用户定义特征变量的边界约束和线性/非线性约束。
    *   **实时可视化**: 在优化过程中，实时绘制收敛曲线和参数空间探索图，让用户可以监控优化进程。
    *   **线程安全**: 所有计算密集型优化任务都在独立的 `QThread` 中运行，保证了GUI的流畅响应，并支持中断操作。

*   **核心类**: `TargetOptimizationModule(QWidget)`, `OptimizationWorker(QThread)`

---

### 🔄 模块七：多目标优化

当需要同时优化多个相互冲突的目标（例如，同时提高强度和韧性）时，此模块可以找到一系列代表“最佳权衡”的帕累托前沿（Pareto Front）解。

*   **核心功能**:
    *   **强大的优化库**: 基于 `pymoo` 库，支持 `NSGA-II`, `NSGA-III`, `SPEA2`, `MOEA/D` 等业界领先的多目标优化算法。
    *   **混合变量处理**: 内置自定义的采样（`MixedVariableSampling`）和修复（`EnhancedBoundaryRepair`）算子，原生支持连续、整数、二元和分类等混合类型的特征变量。
    *   **鲁棒性与约束处理**: 提供了鲁棒优化（`RobustOptimizationHandler`）和显式约束处理（`ExplicitConstraintHandler`）等高级功能。
    *   **实时监控与可视化**: 通过自定义的回调函数（`OptimizationCallback`）实时更新帕累托前沿图和收敛指标（如超体积）。
    *   **检查点机制**: 支持在优化过程中自动保存检查点（`OptimizationCheckpoint`），便于从中断处恢复。

*   **核心类**: `MultiObjectiveOptimizationModule(QWidget)`, `MLProblem(Problem)`

---

### 🤖 模块八：主动学习（实验性 / 非 GUI）

当前仓库提供了一个可复用的主动学习核心算法实现（`modules/active_learning.py` 与 `modules/active_learning/core.py`），用于“推荐-反馈-再学习”的闭环实验设计；它目前**未接入** `ui/main_window.py` 的标签页工作流。

*   **核心能力**:
    *   **数据存储**: `DataStore` 管理已完成实验的 (X, y)。
    *   **设计空间**: 通过 `FeatureDimension`/`DesignSpace` 定义特征范围并生成候选点。
    *   **候选点评分与选择**: 通过采集函数对候选点排序并选择下一批建议实验。
    *   **迭代编排**: `ActiveLoop` 组织完整的主动学习迭代流程（由二次开发者接入具体实验/仿真接口）。

*   **使用方式**: 作为 Python 模块在脚本/Notebook 中调用（以文件内 docstring 与源码为准）；后续可再接入 GUI 形成完整闭环。

---

### 💡 模块九：智能向导

此模块充当用户的智能助手，通过自动化的数据分析，为后续的预处理和建模步骤提供智能化的配置建议。

*   **核心功能**:
    *   **自动化数据分析**: `DataAnalysisWorker` 线程在后台对数据进行全面的统计和质量评估。
    *   **多维度质量评估**: 从完整性、唯一性、有效性、一致性和相关性等多个维度对数据质量进行打分。
    *   **智能推荐**: 基于分析结果，自动推荐最佳的缺失值处理方法、异常值检测策略、特征选择方法和模型选择。
    *   **一键应用**: 用户可以一键采纳向导的推荐配置，并将其自动应用到后续的相关模块中。

*   **核心类**: `IntelligentWizard(QWidget)`, `DataAnalysisWorker(QThread)`

---

### ⚙️ 性能监控

这是一个实时监控工具，用于跟踪应用程序的系统资源占用和后台任务的执行进度。

*   **核心功能**:
    *   **系统资源监控**: `SystemMonitorWorker` 利用 `psutil` 库实时监控 CPU、内存、磁盘和网络的使用情况。
    *   **任务进度跟踪**: `TaskProgressTracker` 能够跟踪长时间运行的任务（如特征选择、模型训练）的进度、耗时和状态。
    *   **实时图表**: 将系统性能数据以实时更新的图表形式展示，便于用户了解系统负载。
    *   **性能警报**: 当CPU或内存使用率过高时，会发出警报，提醒用户可能存在的性能瓶颈。

*   **核心类**: `PerformanceMonitor(QWidget)`, `SystemMonitorWorker(QThread)`

---

### 🤝 协作与版本控制

此模块为团队协作和实验管理提供了基础支持。

*   **核心功能**:
    *   **项目管理**: 支持创建、打开和删除项目，每个项目都有独立的目录结构来存放数据、模型和结果。
    *   **版本快照**: `VersionControl` 类允许用户为项目的当前状态（包括数据和模型）创建一个带有描述信息的“快照”。
    *   **历史追溯**: 所有版本快照都会被记录下来，用户可以随时查看和回顾之前的实验状态。
    *   **项目导出**: 支持将整个项目打包成一个 `.zip` 文件，方便分享和迁移。

*   **核心类**: `CollaborationWidget(QWidget)`, `ProjectManager`, `VersionControl`

---

### ✅ 模块十二：模型预测与结果导出

此模块用于加载已经训练好的模型，对新数据进行预测，并导出结果。

*   **核心功能**:
    *   **模型加载**: 支持加载当前会话中训练的模型，或从 `.joblib` 文件加载持久化的模型。
    *   **详细的模型信息**: 加载模型后，会自动解析并显示模型的类型、参数、特征名称、预处理步骤等详细元数据。
    *   **双重预测模式**:
        *   **批量预测**: 支持从文件或剪贴板导入批量数据进行预测。
        *   **手动输入**: 自动根据模型的特征生成输入界面，用户可以手动输入单一样本的特征值进行实时预测。
    *   **结果导出**: 支持将带有预测结果的数据导出为 CSV 或 Excel 文件。

*   **核心类**: `PredictionModule(QWidget)`

---

### 🧩 模块十三：智能聚类分析

该模块提供一个完整的聚类分析工作流（数据审计 → 特征工程 → 多算法聚类 → 评估 → 可视化 → 报告导出），适合用于材料数据的无监督探索。

*   **核心功能**:
    *   **多算法支持**: K-Means、DBSCAN、Agglomerative、Spectral、BIRCH、MeanShift、GMM、OPTICS 等，并支持可选的 HDBSCAN。
    *   **实时预览**: 调参时可进行实时预览与评估指标计算。
    *   **智能推荐**: 根据数据特征推荐算法与关键参数（如 K 值、DBSCAN eps/min_samples）。
    *   **大数据优化**: 提供面向大规模数据的性能优化与分阶段计算策略。

*   **核心类**: `IntelligentClusteringModule(QWidget)`（见 `modules/clustering/`）
*   **可选依赖**: `umap-learn`（降维）、`hdbscan`（密度聚类与 DBCV 指标）

---

### 🧱 模块十四：ALIGNN 图神经网络（可选）

ALIGNN（Atomistic Line Graph Neural Network）用于基于晶体结构的性质预测，模块实现与 UI 集成位于 `modules/alignn/`，主窗口在缺少依赖时会自动降级为“Unavailable”提示。

*   **数据格式**: CSV/Excel，包含 `id`、目标列、以及 `atoms` 列（JARVIS Atoms 字典 JSON 形式；详见 `modules/alignn/README.md`）
*   **关键依赖**: `torch`、`dgl`、`jarvis-tools`（以及 `pydantic`/`tqdm` 等，详见模块 README）

---

### 💎 模块十五：CGCNN 图神经网络（可选）

CGCNN（Crystal Graph Convolutional Neural Networks）用于基于 CIF 晶体结构的性质预测，模块位于 `modules/cgcnn/`，缺少依赖时同样会自动降级提示。

*   **数据格式**: 数据目录包含 `*.cif`、`id_prop.csv`（两列：`cif_id,target`）、`atom_init.json`（元素特征字典）
*   **关键依赖**: `torch`、`pymatgen`

### 🚀 推荐工作流 (Recommended Workflow) 

为了充分利用 MatSci-ML Studio 的全部功能，我们推荐用户遵循以下工作流程。这个流程与主界面上模块标签页的排列顺序基本一致，旨在引导您从数据到洞见的完整旅程。

1.  **数据导入与探索 (在 `Data Management` 模块)**
    *   **起点**: 一切从这里开始。使用 `Browse` 或 `From Clipboard` 按钮导入您的原始数据集。
    *   **探索**: 在右侧的 "Data Overview" 和 "Data Quality" 标签页中，仔细查看数据的基本信息、预览数据内容、检查缺失值和数据分布情况。
    *   **初步清洗**: 使用 "Data Cleaning" 部分的功能处理明显的缺失值和重复数据。
    *   **定义任务**: 在 "Feature & Target Selection" 部分，选择您的特征 (X) 和目标 (y)。系统将自动推荐任务类型（分类/回归）。
    *   **编码 (可选)**: 如果数据中包含非数值类型的分类特征，使用 "Categorical Feature Encoding" 功能进行编码。

2.  **数据深度处理 (在 `Advanced Preprocessing` 或 `Intelligent Wizard` 模块)**
    *   **智能向导**: 如果不确定如何处理数据，可以切换到 `Intelligent Wizard` 模块，运行智能分析，它会为您提供一套完整的预处理和建模建议。
    *   **手动高级处理**: 或者，在 `Advanced Preprocessing` 模块中，您可以手动执行更高级的操作，如使用多种高级算法检测和处理异常值。

3.  **特征筛选 (在 `Feature Selection` 模块)**
    *   **降维**: 将经过预处理的数据载入此模块。通过配置多阶段的特征选择策略（如重要性过滤、相关性过滤和高级搜索），剔除冗余或无关的特征，以提升模型性能。

4.  **模型训练 (在 `Model Training` 模块)**
    *   **核心步骤**: 使用筛选后的特征集来训练机器学习模型。
    *   **选择模型**: 根据您的任务类型，选择一个或多个候选模型。
    *   **超参数优化 (可选)**: 启用 HPO (Hyperparameter Optimization) 来自动寻找模型的最佳参数组合。
    *   **评估**: 模型训练完成后，在右侧的 "Metrics"、"Confusion Matrix" 等标签页中全面评估模型性能。
    *   **保存模型**: 对结果满意后，点击 `Save Model` 将训练好的完整管道保存下来。

5.  **模型应用与探索 (多模块并行)**
    *   一旦有了训练好的模型，您可以根据需求进入以下任何一个模块：
    *   **模型预测 (`Prediction`)**: 加载模型，对新的未知数据进行预测。
    *   **模型解释 (`SHAP Analysis`)**: 加载模型和数据，深入理解模型为什么会做出这样的预测，分析特征的影响力。
    *   **目标优化 (`Target Optimization` / `Multi-Objective Optimization`)**: 如果您的目标是“反向设计”（即寻找能产生特定性能的配方或参数），请使用这两个模块。加载模型，设定您的优化目标和约束条件，算法将为您搜索最优的特征组合。
    *   **主动学习 (`Active Learning`)**: 如果您的目标是“指导下一次实验”，请使用此模块。它会基于当前模型，推荐最具有信息价值的“下一个实验点”，帮助您用最少的实验次数达到目标。

6.  **项目管理与监控 (贯穿始终)**
    *   **协作与版本控制 (`Collaboration`)**: 在项目的任何关键阶段，您都可以使用此模块创建“快照”，保存当前的工作状态，便于版本控制和团队分享。
    *   **性能监控 (`Performance Monitor`)**: 在执行任何计算密集型任务（如训练、优化）时，可以随时切换到此模块查看系统资源占用情况。

---

### 🛠️ 核心工具与辅助模块 (Core Utilities & Helper Modules)

除了用户直接交互的GUI模块外，本项目还包含一系列在后台提供支撑的核心工具和辅助模块。这部分主要面向希望理解项目底层逻辑或进行二次开发的开发者。

*   **`main_window.py`**
    *   **角色**: 应用程序的主入口和框架。
    *   **职责**: 负责创建主窗口，初始化并容纳所有功能模块的标签页 (`QTabWidget`)，并建立各个模块之间的信号-槽连接（`connect_modules`），确保数据和状态可以在不同模块间顺畅流动。

*   **`ml_utils.py`**
    *   **角色**: 机器学习的“核心大脑”。
    *   **职责**: 封装了所有底层的机器学习逻辑。它定义了平台支持的`sklearn`模型库 (`get_available_models`)，提供默认的超参数范围 (`get_default_hyperparameters`)，并包含模型评估 (`evaluate_*_model`) 和预处理管道创建 (`create_preprocessing_pipeline`) 等核心函数。

*   **`data_utils.py` & `plot_utils.py`**
    *   **角色**: 数据和可视化的基础工具库。
    *   **职责**: `data_utils.py` 封装了所有数据导入和基础质量检查的功能。`plot_utils.py` 则提供了一系列标准化的绘图函数，确保整个应用的可视化风格统一且代码可复用。

*   **`feature_name_utils.py` & `prediction_utils.py`**
    *   **角色**: 特征名称映射与管理系统。
    *   **职责**: 这是处理复杂特征（特别是经过独热编码后的特征）的关键。`FeatureNameMapper` 类负责跟踪原始特征与其编码后名称之间的映射关系。这使得SHAP分析和手动预测等模块依然能够使用人类可读的原始特征名，极大地提升了用户体验和结果的可解释性。

*   **`shap_visualization_windows.py`**
    *   **角色**: SHAP可视化的独立渲染解决方案。
    *   **职责**: 为了解决 `matplotlib` 与 `PyQt` 在某些环境下（特别是在后台线程中）的兼容性问题，该文件定义了独立的 `QDialog` 窗口。当用户需要查看SHAP图时，程序会创建一个新的独立窗口来渲染和显示图像，确保了主界面的稳定性和响应性。

*   **`EnhancedVisualizations` (在 `active_learning_optimizer.py` 中)**
    *   **角色**: 主动学习模块的专属可视化工具。
    *   **职责**: 虽然不是一个顶层模块，但它为 `ActiveLearningWindow` 提供了多种高级的可视化图表，如PCA降维分析、不确定性分析、设计空间切片和优化历史等，是主动学习模块能够提供深度洞察的关键。


## 🚀 快速开始

### 安装依赖
- 将包导入到python工具当中，例如pycharm等，推荐python版本3.13+
```bash
pip install PyQt5 numpy pandas scikit-learn matplotlib seaborn joblib scipy openpyxl xlrd
```
- 如果上述按照失败，请手动按照如下命令安装相应的依赖
```bash
pip install PyQt5 numpy pandas scikit-learn matplotlib seaborn joblib scipy openpyxl xlrd
pip install shap psutil xgboost lightgbm catboost pymoo scikit-optimize umap-learn hdbscan
```
### 启动应用程序

```bash
python main.py
```



## 🔧 系统要求

- Python 3.10
- PyQt5 5.15.0+
- 科学计算库：NumPy、Pandas、SciPy
- 机器学习库：scikit-learn、XGBoost、LightGBM、CatBoost
- 可视化库：Matplotlib、Seaborn
- 模型解释库：SHAP



## 📞 联系方式

如有问题或建议，请联系：1255201958@qq.com 


---



## 目录

1.  [**项目概述 (Introduction)**](#1-项目概述-introduction)
    *   [1.1. 项目愿景与目标](#11-项目愿景与目标)
    *   [1.2. 核心功能亮点](#12-核心功能亮点)
    *   [1.3. 目标用户](#13-目标用户)

2.  [**架构与工作流 (Architecture & Workflow)**](#2-架构与工作流-architecture--workflow)
    *   [2.1. 模块化架构](#21-模块化架构)
    *   [2.2. 标准工作流](#22-标准工作流)
    *   [2.3. 高级工作流](#23-高级工作流)
    *   [2.4. 核心技术栈](#24-核心技术栈)

3.  [**环境搭建与依赖 (Setup & Dependencies)**](#3-环境搭建与依赖-setup--dependencies)
    *   [3.1. 核心依赖](#31-核心依赖)
    *   [3.2. 可选高级依赖](#32-可选高级依赖)
    *   [3.3. 安装指南](#33-安装指南)

4.  [**模块深度解析 (Module Deep Dive)**](#4-模块深度解析-module-deep-dive)
    *   [4.1. 主窗口 (`main_window.py`)](#41-主窗口-main_windowpy)
    *   [4.2. 数据管理模块 (`data_module.py`)](#42-数据管理模块-data_modulepy)
    *   [4.3. 智能向导 (`intelligent_wizard.py`)](#43-智能向导-intelligent_wizardpy)
    *   [4.4. 高级预处理 (`advanced_preprocessing.py`)](#44-高级预处理-advanced_preprocessingpy)
    *   [4.5. 特征工程与选择 (`feature_module.py`)](#45-特征工程与选择-feature_modulepy)
    *   [4.6. 模型训练与评估 (`training_module.py`)](#46-模型训练与评估-training_modulepy)
    *   [4.7. 主动学习与优化 (`active_learning_optimizer.py` & `active_learning_window.py`)](#47-主动学习与优化-active_learning_optimizerpy--active_learning_windowpy)
    *   [4.8. 多目标优化 (`multi_objective_optimization.py`)](#48-多目标优化-multi_objective_optimizationpy)
    *   [4.9. 模型解释性 (SHAP) (`shap_analysis.py`)](#49-模型解释性-shap-shap_analysispy)
    *   [4.10. 目标优化 (`target_optimization.py`)](#410-目标优化-target_optimizationpy)
    *   [4.11. 模型预测 (`prediction_module.py`)](#411-模型预测-prediction_modulepy)
    *   [4.12. 协作与版本控制 (`collaboration_version_control.py`)](#412-协作与版本控制-collaboration_version_controlpy)
    *   [4.13. 性能监控 (`performance_monitor.py`)](#413-性能监控-performance_monitorpy)
    *   [4.14. 辅助工具模块 (Utils & Others)](#414-辅助工具模块-utils--others)

5.  [**技术实现与设计哲学 (Technical Implementation & Philosophy)**](#5-技术实现与设计哲学-technical-implementation--philosophy)
    *   [5.1. 异步处理与多线程](#51-异步处理与多线程)
    *   [5.2. 数据流与信号/槽机制](#52-数据流与信号槽机制)
    *   [5.3. 鲁棒性与错误处理](#53-鲁棒性与错误处理)
    *   [5.4. 特征名管理 (`feature_name_utils.py`)](#54-特征名管理-feature_name_utilspy)

6.  [**未来展望与贡献 (Future Work & Contribution)**](#6-未来展望与贡献-future-work--contribution)

---

## 1. 项目概述 (Introduction)

### 1.1. 项目愿景与目标

**MatSci-ML Studio** 是一款专为材料科学（MatSci）领域设计的、智能化的、端到端的机器学习（ML）集成开发环境（IDE）。项目的核心愿景是**降低材料研发人员应用机器学习技术的门槛，加速新材料的发现与性能优化过程**。

传统的材料研发流程严重依赖于昂贵且耗时的“试错法”实验。机器学习，特别是主动学习和多目标优化，为这一领域带来了革命性的变革。然而，一个完整的机器学习工作流——从数据清洗、特征工程到模型训练、超参数优化，再到模型解释和实验设计——涉及大量复杂的步骤和专业知识。

MatSci-ML Studio 旨在将这一复杂流程封装在一个图形化、模块化的交互界面中，实现以下核心目标：

*   **工作流自动化**：通过模块化的设计，引导用户完成从数据到洞见的完整流程。
*   **智能化引导**：内置的“智能向导”系统能够分析数据，并为预处理、模型选择和超参数调优提供专业建议。
*   **前沿算法集成**：集成了主动学习、多目标优化、SHAP模型解释等先进算法，让非算法专家也能利用最前沿的技术指导实验设计。
*   **可视化与可解释性**：提供丰富的、交互式的可视化工具，不仅展示结果，更致力于解释“为什么”，增强用户对模型和数据的理解与信任。
*   **协作与可复现性**：通过内置的项目管理和版本控制系统，确保研究过程的可追溯性和团队协作的便利性。

### 1.2. 核心功能亮点

*   **端到端工作流**：覆盖从数据导入、预处理、特征选择、模型训练、性能评估、模型解释到最终预测和优化的完整机器学习生命周期。
*   **智能向导系统**：自动分析数据质量和特征，为用户推荐最佳的预处理策略、模型选择和超参数配置。
*   **高级数据预处理**：提供异常值检测（IQR, Z-Score, Isolation Forest等）、智能缺失值填补（KNN, 迭代填补等）以及多种数据变换和缩放方法。
*   **多策略特征选择**：结合过滤法、封装法和嵌入法，提供基于模型重要性、相关性分析以及遗传算法等高级搜索策略的特征选择方案。
*   **全面的模型库与训练**：支持包括Scikit-learn、XGBoost、LightGBM、CatBoost在内的多种回归和分类模型，并提供网格搜索、随机搜索和贝叶斯搜索等超参数优化（HPO）方法。
*   **深度模型可解释性 (SHAP)**：集成SHAP（SHapley Additive exPlanations）框架，提供全局和局部的模型解释，包括特征重要性、摘要图、依赖图和瀑布图，帮助用户理解模型决策依据。
*   **主动学习与贝叶斯优化**：内置强大的主动学习引擎（`ActiveLearningOptimizer`），通过贝叶斯优化和多种采集函数（如预期提升EI、置信上界UCB）智能推荐下一个最有价值的实验点，极大提升研发效率。
*   **多目标优化 (Pareto)**：支持同时优化多个（≥2）材料性能指标，利用NSGA-II等遗传算法寻找帕累托最优解集，直观展示性能间的权衡（Trade-off）。
*   **项目管理与版本控制**：提供项目创建、管理、快照（版本）创建功能，确保实验的可复现性和团队成员间的无缝协作。
*   **实时性能监控**：监控CPU、内存等系统资源使用情况，并在高负载时提供预警，保证大型计算任务的稳定性。

### 1.3. 目标用户

*   **材料科学家/工程师**：希望利用机器学习加速研发，但缺乏深厚编程或算法背景的研究人员。
*   **计算材料学研究者**：需要一个集成化平台来快速验证想法、训练模型和进行数据分析的科研人员。
*   **数据科学家**：在材料领域工作，需要一个高效、标准化的工具来处理和建模材料数据。
*   **学生与教育者**：用于教授和学习应用机器学习在科学研究中的实际案例。

## 2. 架构与工作流 (Architecture & Workflow)

### 2.1. 模块化架构

MatSci-ML Studio 采用高度模块化的架构，每个核心功能被封装在一个独立的`module`中。这种设计带来了极佳的可维护性、可扩展性和用户友好性。

**核心模块关系图:**
```
[ 用户 ]
   |
   V
[ main.py (主应用框架) ]
   |
   +--- [ collaboration_version_control.py (项目/版本管理) ]
   |
   +--- [ performance_monitor.py (系统性能监控) ]
   |
   +--- [ intelligent_wizard.py (智能向导) ]
   |
   +--- [ Tabbed Workflow Interface ]
         |
         +--> 1. data_module.py (数据导入与探索)
         |      | (data_ready signal)
         |      V
         +--> 2. advanced_preprocessing.py (高级预处理)
         |      | (preprocessing_completed signal)
         |      V
         +--> 3. feature_module.py (特征选择)
         |      | (features_ready signal)
         |      V
         +--> 4. training_module.py (模型训练)
                | (model_ready signal)
                |
                +---------------------------------+
                |                                 |
                V                                 V
         +--> 5. prediction_module.py (预测)    +--> 8. active_learning_window.py (主动学习)
         |                                        |
         +--> 6. shap_analysis.py (模型解释)     +--> 9. multi_objective_optimization.py (多目标优化)
         |
         +--> 7. target_optimization.py (目标优化)
```

*   **`main_window.py`** 是应用程序的入口和主框架，负责初始化所有模块并将它们组织在标签页中。它还处理菜单栏、状态栏和模块间的信号/槽连接。
*   **辅助模块**（`collaboration`, `performance_monitor`, `intelligent_wizard`）提供系统级功能，与核心工作流并行存在。
*   **核心工作流模块**（`data_module` 到 `prediction_module` 等）通过 **PyQt的信号/槽机制** 实现解耦和数据传递。例如，当`DataModule`准备好数据后，会发射一个`data_ready`信号，`FeatureModule`接收此信号并加载数据，以此类推。这种设计使得流程清晰，且易于修改或替换单个模块。

### 2.2. 标准工作流

一个典型的材料性能预测任务会遵循以下步骤：

1.  **项目创建**：在`Collaboration`模块中创建一个新项目，为所有数据、模型和结果提供一个独立的存储空间。
2.  **数据加载**：在`Data Management`模块中，导入包含材料组分、工艺参数和性能指标的`.csv`或`.xlsx`文件。该模块提供数据预览、统计信息和基本的数据质量可视化。
3.  **数据预处理**：切换到`Advanced Preprocessing`模块，根据智能推荐或手动选择，处理缺失值、检测并处理异常值、进行数据类型转换。每一步操作都可以通过`State Management`进行撤销/重做。
4.  **特征选择**：处理后的数据自动传递到`Feature Selection`模块。在这里，用户可以配置多阶段的特征选择策略，例如先用模型重要性进行初筛，再用相关性分析剔除冗余特征，最后使用遗传算法等进行高级子集搜索。
5.  **模型训练**：优选出的特征集被传递到`Model Training`模块。用户选择一个或多个机器学习模型（如随机森林、XGBoost），配置训练参数（如测试集比例、交叉验证折数），并可选择性地进行超参数优化。训练完成后，模块会展示详细的性能评估指标（R², MAE, 混淆矩阵, ROC曲线等）。
6.  **模型解释与预测**：
    *   训练好的模型可以传递到`SHAP Analysis`模块，进行深度可解释性分析，理解模型决策的内部逻辑。
    *   模型也可以传递到`Prediction Module`，用于对新的、未知的候选材料进行性能预测。
7.  **结果导出与保存**：在每个阶段，用户都可以导出数据、图表、模型和报告。最后，通过`File -> Save Project`保存整个工作流的状态。

### 2.3. 高级工作流 (优化与设计)

当目标不是预测，而是**发现新材料**时，用户可以采用更高级的工作流：

1.  **主动学习 (Active Learning)**：
    *   在`Active Learning`模块中，加载初始的小型训练集和大型的、未标记的候选材料库（虚拟数据集）。
    *   运行分析后，系统会基于贝叶斯优化，推荐一个或一批最有信息价值的候选材料。
    *   用户根据推荐进行实验，获得真实性能数据后，通过`Iterative Feedback`界面将新数据点加入训练集。
    *   重复此过程，系统会用最少的实验次数，快速逼近性能最优的材料。

2.  **多目标优化 (Multi-Objective Optimization)**：
    *   用户在`Multi-objective Optimization`模块中加载两个或多个预训练好的模型，每个模型预测一个不同的性能指标（例如，一个模型预测强度，另一个预测韧性）。
    *   配置每个特征（材料组分、工艺参数）的取值范围和约束条件。
    *   启动优化后，系统使用NSGA-II等遗传算法，在巨大的设计空间中搜索，并最终给出一系列**帕累托最优解**。
    *   结果以帕累托前沿图的形式展示，直观地揭示了不同性能指标之间的权衡关系，为用户提供了一系列“两全其美”的候选方案。

### 2.4. 核心技术栈

*   **GUI框架**: `PyQt5` - 成熟、功能强大的跨平台GUI工具包。
*   **数据处理**: `pandas`, `numpy` - Python数据科学生态系统的基石。
*   **机器学习**: `scikit-learn` - 提供丰富的预处理、特征选择和建模算法。
*   **高级模型**: `xgboost`, `lightgbm`, `catboost` - 集成了业界领先的梯度提升树模型。
*   **优化算法**: `pymoo` - 用于多目标遗传算法；`scikit-optimize` - 用于贝叶斯优化。
*   **模型解释**: `shap` - 提供最先进的模型可解释性分析。
*   **可视化**: `matplotlib`, `seaborn` - 强大的静态和统计图表库。
*   **系统监控**: `psutil` - 用于实时监控系统资源。

## 3. 环境搭建与依赖 (Setup & Dependencies)

为了成功运行 AutoMatFlow，需要安装以下Python库。建议使用`conda`或`venv`创建独立的虚拟环境。

### 3.1. 核心依赖
这些是运行基本功能所必需的库。
```bash
pip install PyQt5
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install joblib
```

### 3.2. 可选高级依赖
这些库用于启用高级功能，如梯度提升模型、优化算法和模型解释。
```bash
# 梯度提升模型
pip install xgboost
pip install lightgbm
pip install catboost

# 优化算法
pip install pymoo
pip install scikit-optimize

# 模型解释
pip install shap

# 系统监控
pip install psutil
```

### 3.3. 安装指南
建议使用 `pip` 一次性安装所有依赖：
```bash
# 创建并激活虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 安装所有依赖
pip install PyQt5 pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm catboost pymoo scikit-optimize shap psutil
```

## 4. 模块深度解析 (Module Deep Dive)

本节将逐一深入分析每个模块的实现细节、核心类和关键方法。

### 4.1. 主窗口 (`main.py`)
*   **角色**: 应用程序的入口和总指挥。
*   **核心类**: `MatSciMLStudioWindow(QMainWindow)`
*   **功能**:
    *   `init_ui()`: 初始化主窗口、菜单栏、状态栏和标签页（`QTabWidget`）。
    *   `create_menu_bar()`: 构建复杂的菜单系统，提供项目管理、数据导入/导出、模块导航和帮助等功能。
    *   `setup_modules()`: 实例化所有功能模块（如 `DataModule`, `TrainingModule` 等），并将它们添加为`QTabWidget`的标签页。这是模块化架构的核心实现。
    *   `connect_modules()`: **关键方法**。通过`pyqtSignal`和`pyqtSlot`建立模块间的数据流。例如，`self.data_module.data_ready.connect(self.feature_module.set_data)`，确保了当数据准备好后，能自动传递给下一个模块。这种松耦合的设计使得模块替换和功能扩展变得容易。
    *   `_collect_project_state()` & `_restore_project_state()`: 实现了项目的保存和加载功能。通过遍历所有模块，收集其当前状态（如加载的文件路径、选择的参数等），并将其序列化为JSON。加载时则反向操作，恢复各模块的状态。
    *   `update_status()` & `update_progress()`: 提供全局的状态和进度更新机制，供所有子模块调用。

### 4.2. 数据管理模块 (`data_module.py`)
*   **角色**: 数据进入系统的第一站，负责数据导入、探索和基础清洗。
*   **核心类**: `DataModule(QWidget)`
*   **功能**:
    *   **数据导入**: 支持从`.csv`、`.xlsx`文件和系统剪贴板导入数据。UI上提供了对分隔符、编码、表头行等参数的配置。
    *   **数据探索**: 导入后，模块会自动生成数据概览：
        *   `Data Information`: 显示数据维度、列名、数据类型、非空值数量和唯一值数量。
        *   `Data Preview`: 在`QTableWidget`中展示前100行数据。
        *   **交互式可视化**: 自动生成多个可视化标签页，包括：
            *   `Data Quality`: 缺失值热力图和条形图。
            *   `Statistical Summary`: `df.describe()`的详细统计摘要。
            *   `Distributions`: 用户可选择任意特征，查看其直方图、箱线图或饼图。
            *   `Correlations`: 数值特征间的相关性热力图。
    *   **数据清洗**: 提供简单的清洗功能，如按行/列丢弃缺失值、填充（均值、中位数、众数）、移除重复行。
    *   **特征/目标选择**: 用户通过`QListWidget`和`QComboBox`选择特征（X）和目标（y）。系统会根据目标列的特性（`suggest_task_type`）自动推荐任务类型（分类/回归）。
    *   **分类特征编码**: 内置了对分类特征的编码功能（One-Hot, Label Encoding），并智能地将目标列排除在编码之外，防止数据泄露。
    *   **数据传递**: 当用户确认选择后，模块会发射`data_ready(X, y)`信号，将处理好的特征矩阵和目标向量传递给后续模块。

### 4.3. 智能向导 (`intelligent_wizard.py`)
*   **角色**: 一个自动化的数据分析师和配置顾问。
*   **核心类**: `IntelligentWizard(QWidget)`, `DataAnalysisWorker(QThread)`
*   **功能**:
    *   **异步分析**: 当用户点击“Start Intelligent Analysis”时，`DataAnalysisWorker`会在后台线程中对数据进行全面分析，避免UI卡顿。
    *   **数据深度分析 (`_analyze_data_statistics`, `_assess_data_quality`)**:
        *   统计样本量、特征数、特征类型分布。
        *   评估数据质量，包括缺失数据比例、异常值（使用IQR方法初筛）、重复行和高相关性特征对。
        *   计算一个总体的**数据质量分**。
    *   **任务类型推断 (`_infer_task_type`)**: 根据目标变量`y`的类型和唯一值数量，自动判断是分类任务还是回归任务。
    *   **智能推荐 (`_generate_recommendations`)**: 这是向导的核心。基于上述分析结果，生成一个包含多方面建议的配置字典：
        *   **预处理建议**: 如“数据缺失率高，建议使用高级插值法”。
        *   **特征选择建议**: 如“特征数量>50，建议启用基于重要性的过滤”。
        *   **模型推荐**: 根据任务类型和数据规模，推荐合适的模型列表（如小数据集推荐SVM，大数据集推荐LightGBM）。
        *   **超参数建议**: 推荐交叉验证的折数、HPO的搜索方法（大数据用随机搜索，小数据用网格搜索）和迭代次数。
        *   **工作流建议**: 给出一个推荐的、从头到尾的操作步骤。
    *   **配置应用**: 用户可以在UI上选择接受部分或全部建议。点击“Apply”后，`configuration_ready`信号被发射，主窗口接收此信号并将配置应用到相应的模块中。

### 4.4. 高级预处理 (`advanced_preprocessing.py`)
*   **角色**: 提供比`DataModule`更深入、更专业的数据清洗和变换功能。
*   **核心类**:
    *   `AdvancedPreprocessing(QWidget)`: 主UI界面。
    *   `OutlierDetector`: 封装了多种异常值检测算法。
    *   `SmartImputer`: 封装了多种高级缺失值填补算法。
    *   `StateManager`: 实现了撤销/重做功能的状态机。
*   **功能**:
    *   **异常值处理**:
        *   **检测**: 提供IQR、Z-Score，以及更高级的基于模型的检测方法，如**Isolation Forest**、**Local Outlier Factor (LOF)** 和 **One-Class SVM**。
        *   **处理**: 用户可以选择移除异常值所在的行、用NaN替换、用均值/中位数替换，或者进行**盖帽处理**（将异常值拉回至1%或99%分位数）。
        *   **可视化**: 提供箱线图等可视化工具，直观展示异常值的分布。
    *   **智能缺失值填补**:
        *   除了基本的均值/中位数填充，还提供了**KNN Imputer**（使用K近邻样本的特征来预测缺失值）和**Iterative Imputer**（使用其他特征作为输入，通过回归模型来预测缺失值），这对于处理复杂的缺失模式非常有效。
    *   **数据变换**:
        *   提供多种数据缩放器（`StandardScaler`, `MinMaxScaler`, `RobustScaler`），以及更高级的分布变换工具（`PowerTransformer`, `QuantileTransformer`），用于处理偏态分布的数据，使其更接近正态分布，这对于很多线性模型和神经网络至关重要。
    *   **操作历史与状态管理**:
        *   `StateManager`会记录每一次数据处理操作（如“应用IQR异常值检测”），并将数据快照（`DataState`）入栈。
        *   用户可以通过“Undo”和“Redo”按钮在不同的数据状态间切换，极大地增强了数据处理的灵活性和容错性。

### 4.5. 特征工程与选择 (`feature_module.py`)
*   **角色**: 从原始特征中筛选出对模型最有用的特征子集，是提升模型性能和降低计算复杂度的关键步骤。
*   **核心类**: `FeatureModule(QWidget)`, `FeatureSelectionWorker(QThread)`
*   **功能**:
    *   **多阶段选择策略**: 模块的设计遵循一个逻辑清晰的三步筛选流程：
        1.  **重要性过滤**: 使用一个基准模型（如随机森林）快速计算所有特征的重要性得分，并根据用户设置的规则（如保留Top K个特征、保留重要性高于阈值的特征、或保留累计重要性达到95%的特征）进行第一轮粗筛。
        2.  **相关性过滤**: 计算剩余特征间的皮尔逊相关系数矩阵，移除相关性过高（如>0.95）的特征对中的一个。为了智能地决定保留哪一个，模块提供了两种策略：a) 保留与目标变量相关性更高的那个；b) 移除后模型性能更优的那个。
        3.  **高级子集搜索 (可选)**: 对经过前两步筛选后的特征子集，进行更精细的搜索，以找到最优的特征组合。提供了三种高级搜索算法：
            *   **序列特征选择 (SFS)**: 包括前向选择（SFS）和后向消除（SBS）。
            *   **遗传算法 (GA)**: 模拟生物进化过程，通过交叉、变异和选择来搜索最优特征子集，适合中等规模的特征空间。
            *   **穷举搜索 (Exhaustive Search)**: 暴力搜索所有可能的特征组合，能找到理论最优解，但仅适用于特征数非常少（≤15）的情况。
    *   **异步执行**: 所有的特征选择计算都在`FeatureSelectionWorker`后台线程中执行，并通过`progress_updated`和`status_updated`信号实时向主UI报告进度，避免了界面冻结。
    *   **模型驱动评估**: 所有选择策略都依赖于一个用户选定的评估模型。模块会反复使用交叉验证来评估不同特征子集的性能，确保选出的特征是真正对模型有益的。
    *   **丰富的可视化**:
        *   展示特征重要性条形图。
        *   对比展示相关性过滤前后的热力图。
        *   如果使用遗传算法，会展示适应度（模型性能）随代数进化的曲线。

### 4.6. 模型训练与评估 (`training_module.py`)
*   **角色**: 机器学习工作流的核心，负责模型训练、超参数优化和全面的性能评估。
*   **核心类**: `TrainingModule(QWidget)`
*   **功能**:
    *   **模型库 (`get_available_models`)**:
        *   动态地根据任务类型（分类/回归）提供一个丰富的模型列表。
        *   除了Scikit-learn的经典模型，还无缝集成了XGBoost, LightGBM, CatBoost等业界领先的框架。
    *   **预处理管道 (`create_preprocessing_pipeline`)**:
        *   **自动化与鲁棒性**: 这是模块设计的精髓之一。它会自动识别数值型、分类型和布尔型特征。
        *   为不同类型的特征创建独立的预处理管道（如数值型进行缺失值填充+标准化，分类型进行独热编码），然后通过`ColumnTransformer`将它们组合成一个统一的预处理步骤。
        *   这个预处理步骤会与最终的模型打包成一个完整的Scikit-learn `Pipeline`。这样做的好处是：
            1.  **防止数据泄露**: 所有预处理步骤（如标准化）的参数都只在训练集上学习，然后应用到测试集。
            2.  **简化部署**: 保存和加载模型时，整个预处理流程和模型被当作一个单一对象，极大地简化了后续的预测流程。
    *   **超参数优化 (HPO)**:
        *   用户可选择启用HPO，并从**网格搜索**、**随机搜索**和**贝叶斯搜索**（通过`scikit-optimize`）中选择。
        *   模块为每个模型提供了预定义的、合理的超参数搜索空间（`get_default_hyperparameters`），用户也可以通过“Configure Parameter Space”按钮进行自定义。
    *   **全面的性能评估**:
        *   **分类任务**: 生成混淆矩阵（原始值和归一化）、ROC曲线、精确率-召回率曲线，并提供包含精确率、召回率、F1分数和支持度的详细分类报告。
        *   **回归任务**: 生成预测值vs真实值散点图、残差图，并计算R², MAE, MSE, RMSE等关键指标。
    *   **信号传递**: 训练完成后，发射`model_ready`信号，将训练好的完整`Pipeline`对象以及特征名、训练数据等元信息传递给`PredictionModule`, `SHAPAnalysis`, `TargetOptimization`等后续模块。

### 4.7. 主动学习与优化 (`active_learning_optimizer.py` & `active_learning_window.py`)
*   **角色**: 核心智能模块之一，旨在通过最少的实验次数找到最优材料配方或工艺参数。
*   **核心类**: `ActiveLearningOptimizer`, `ActiveLearningWindow`, `ActiveLearningSession`
*   **功能**:
    *   **贝叶斯优化框架**:
        *   `ActiveLearningOptimizer`是整个功能的后端引擎。它基于贝叶斯优化理论，即“代理模型 + 采集函数”。
        *   **代理模型 (Surrogate Model)**: 用户可以选择多种模型（如随机森林、高斯过程）来拟合现有的“实验数据-性能”关系，并预测未知点的性能及其**不确定性**。
        *   **采集函数 (Acquisition Function)**: 提供**预期提升 (Expected Improvement, EI)** 和 **置信上界 (Upper Confidence Bound, UCB)** 两种策略。采集函数会综合考虑代理模型的预测值（**利用, Exploitation**）和不确定性（**探索, Exploration**），计算出每个候选点的“信息价值”得分。
    *   **迭代式学习循环**:
        *   `ActiveLearningWindow`是前端UI，它管理着一个`ActiveLearningSession`对象。
        *   **循环开始**: 用户加载初始训练数据和大量候选数据。
        *   **分析与推荐**: 点击“Start Analysis”，`ActiveLearningOptimizer`开始工作，为每个候选点计算采集分数，并推荐得分最高的点。
        *   **实验反馈**: UI会展示推荐的实验参数。用户进行实验后，在`Iterative Feedback`区域输入真实的实验结果。
        *   **数据更新**: 提交结果后，新的数据点被添加到`ActiveLearningSession`中，训练集增大。
        *   **模型更新与再推荐**: 系统自动使用更新后的训练集重新训练代理模型，并推荐下一个实验点。
    *   **丰富的可视化**:
        *   **探索地图**: 以“预测性能”为x轴，“不确定性”为y轴，直观展示模型的探索与利用情况。
        *   **学习曲线**: 实时绘制模型性能（R²）、发现的最佳值、平均不确定性等指标随迭代次数的变化，帮助用户判断学习过程是否收敛。

### 4.8. 多目标优化 (`multi_objective_optimization.py`)
*   **角色**: 解决现实世界中需要在多个冲突性能指标间进行权衡的问题。
*   **核心类**: `MultiObjectiveOptimizationModule(QWidget)`, `MLProblem(Problem)`, `OptimizationWorker(QThread)`
*   **功能**:
    *   **多模型集成**: 允许用户加载多个独立的、预训练好的模型，每个模型对应一个优化目标（如`模型A预测强度`，`模型B预测韧性`）。
    *   **基于遗传算法的搜索**:
        *   **Pymoo集成**: 后端深度集成了强大的`pymoo`库，利用其高效的**NSGA-II**（非支配排序遗传算法II）和**NSGA-III**等算法。
        *   **问题定义 (`MLProblem`)**: 将机器学习预测问题巧妙地封装成`pymoo`可以理解的优化问题。`_evaluate`方法接收一组候选解（即特征向量），并调用所有模型进行批量预测，返回一个多维的目标向量。
    *   **约束处理**:
        *   **混合变量支持**: `MixedVariableSampling` 和 `EnhancedBoundaryRepair` 是两个关键的自定义类，它们确保遗传算法在搜索过程中能正确处理连续型、整型和分类型变量，并始终保持解在用户定义的边界内。
        *   **显式约束**: 支持用户定义线性和非线性约束（如 `A + B <= 1`）。
    *   **帕累托前沿可视化**:
        *   优化的最终结果不是单个解，而是一系列**帕累托最优解**。
        *   模块提供了强大的可视化功能，可以绘制2D或3D的帕累托前沿图，并为更高维度提供平行坐标图，帮助用户直观地理解不同性能目标之间的权衡关系。
    *   **实时监控**: `OptimizationCallback`类会在每一代遗传算法结束后被调用，将种群的适应度、多样性等信息通过信号发送给主UI，实现优化的实时可视化。

### 4.9. 模型解释性 (SHAP) (`shap_analysis.py`)
*   **角色**: 打开机器学习“黑箱”，帮助用户理解模型为什么会做出特定的预测。
*   **核心类**: `SHAPAnalysisModule(QWidget)`, `SHAPCalculationWorker(QThread)`
*   **功能**:
    *   **SHAP框架集成**: 核心是集成了`shap`库，这是目前最主流的模型无关解释性框架。
    *   **智能解释器选择**: 模块能自动根据加载的模型类型（如树模型、线性模型、或其它黑箱模型）选择最高效的SHAP解释器（`TreeExplainer`, `LinearExplainer`, `KernelExplainer`），并对Pipeline模型进行了特殊优化，以避免常见的兼容性问题。
    *   **全局解释**:
        *   **特征重要性图**: 直观展示哪些特征对模型的整体预测贡献最大。
        *   **摘要图 (Summary Plot)**: 提供了更丰富的信息，不仅显示特征重要性，还通过颜色表示特征值的高低，揭示了特征值与SHAP值的正/负相关关系。
    *   **局部解释**:
        *   **瀑布图 (Waterfall Plot)**: 针对单个样本，清晰地展示了从基准值（全体样本的平均预测）出发，每个特征是如何将预测“推高”或“拉低”至最终值的。
        *   **力图 (Force Plot)**: 同样是针对单个样本，以一种“力平衡”的视觉形式展示所有特征的综合作用。
    *   **特征依赖图 (Dependence Plot)**:
        *   展示单个特征的值如何影响其自身的SHAP值，揭示特征与预测之间的非线性关系。
        *   支持选择一个**交互特征**，通过颜色来揭示两个特征之间的交互效应对预测的影响。

### 4.10. 目标优化 (`target_optimization.py`)
*   **角色**: 在`multi_objective_optimization`模块的基础上，提供了一个更专注于**单目标**优化，并集成了多种经典和现代优化算法的界面。
*   **核心类**: `TargetOptimizationModule(QWidget)`, `GeneticAlgorithm`, `ParticleSwarmOptimization`, `OptimizationWorker`
*   **功能**:
    *   **丰富的算法库**:
        *   **启发式算法**: 提供了从头实现的**遗传算法(GA)**和**粒子群优化(PSO)**，并支持混合变量类型（连续、整数、分类）。
        *   **Scipy集成**: 集成了`scipy.optimize`中的经典算法，如**差分进化 (Differential Evolution)**、**盆地跳跃 (Basin Hopping)**和**COBYLA**（处理约束）。
        *   **贝叶斯优化**: 集成了`scikit-optimize`，提供基于高斯过程的贝叶斯优化。
    *   **混合变量优化**: `GeneticAlgorithm`和`ParticleSwarmOptimization`类经过特殊设计，可以同时处理连续型、整型和分类型特征，这在材料科学领域尤为重要。它们使用混合的交叉和变异算子（如SBX+均匀交叉）来适应不同类型的变量。
    *   **约束处理**: 用户可以定义线性和非线性的硬约束和软约束。后端通过罚函数法或COBYLA等原生支持约束的算法来处理这些限制。
    *   **实时交互式可视化**: `OptimizationWorker`在优化过程中会实时地将每一代的最佳值、种群分布（对于GA/PSO）等信息传递给UI，动态绘制收敛曲线和参数空间探索图，让用户可以实时监控优化进程。

### 4.11. 模型预测 (`prediction_module.py`)
*   **角色**: 应用已训练好的模型进行预测的终端。
*   **核心类**: `PredictionModule(QWidget)`
*   **功能**:
    *   **双重预测模式**:
        *   **文件导入模式**: 用户可以加载一个包含大量候选材料的`.csv`或`.xlsx`文件，进行批量预测。
        *   **手动输入模式**: UI会根据模型的特征自动生成一个输入表单。用户可以手动输入一组特征值，进行单点预测。这是“What-if”分析的绝佳工具。
    *   **特征名智能映射**: 这是该模块最关键的技术点。机器学习模型在训练时，分类特征通常会被独热编码（One-Hot Encoding），导致特征名发生变化（如`'材料类型'`变为`'材料类型_A'`, `'材料类型_B'`等）。该模块通过`prediction_utils.py`中维护的`OriginalFeatureMapper`，实现了**从原始特征到编码后特征的自动映射**。这意味着用户在手动输入时，只需与原始、有意义的特征名（如“温度”、“压力”）交互，后端会自动将其转换为模型能理解的、编码后的格式。
    *   **详细模型信息**: 加载模型后，模块会解析模型（即使是Pipeline），展示其类型、参数、预处理步骤等详细信息，增强了透明度。
    *   **结果导出**: 预测结果可以方便地导出为CSV或Excel文件，并附带模型信息。

### 4.12. 协作与版本控制 (`collaboration_version_control.py`)
*   **角色**: 为整个研究工作流提供项目管理和数据溯源能力。
*   **核心类**: `CollaborationWidget(QWidget)`, `ProjectManager`, `VersionControl`
*   **功能**:
    *   **项目管理 (`ProjectManager`)**:
        *   用户可以创建具有名称、描述和作者等元信息的新项目。
        *   每个项目在本地文件系统上都有一个独立的、结构化的目录（包含`data`, `models`, `results`等子目录）。
        *   提供项目列表、查看详情和删除项目的功能。
    *   **版本控制 (`VersionControl`)**:
        *   **快照 (Snapshot)**: 用户可以在任意时间点，为当前项目创建一个“快照”。这会完整地复制当前项目的所有文件（数据、模型、结果）到一个带时间戳的版本目录中。
        *   这实现了类似于Git的commit功能，但更侧重于实验状态的完整备份，确保了任何历史实验结果都可以被精确复现。
    *   **导出与分享**: 支持将整个项目（包括所有版本）打包成一个`.zip`文件，方便与同事分享或归档。

### 4.13. 性能监控 (`performance_monitor.py`)
*   **角色**: 一个系统级的后台服务，用于监控应用的资源消耗。
*   **核心类**: `PerformanceMonitor(QWidget)`, `SystemMonitorWorker(QThread)`
*   **功能**:
    *   **实时系统监控**: `SystemMonitorWorker`利用`psutil`库，在后台线程中持续收集CPU使用率、内存占用、磁盘空间等信息。
    *   **任务进度跟踪**: `TaskProgressTracker`类可以被其他模块调用，用于跟踪长时间运行的任务（如特征选择、模型训练）的进度。
    *   **可视化仪表盘**: UI上以进度条和图表的形式实时展示系统资源和任务进度。
    *   **性能预警**: 当CPU或内存使用率超过阈值（如90%）时，会发出`performance_alert`信号，主窗口可以捕获此信号并向用户发出警告，防止系统因资源耗尽而崩溃。

### 4.14. 辅助工具模块 (Utils & Others)
*   **`data_utils.py`**: 提供了数据加载（CSV, Excel）、质量报告生成、异常值检测等基础数据处理函数，被`DataModule`和`AdvancedPreprocessing`等模块调用。
*   **`ml_utils.py`**: 提供了机器学习相关的核心辅助函数，如获取可用模型列表、获取默认超参数、创建预处理管道、评估模型性能等，被`FeatureModule`和`TrainingModule`广泛使用。
*   **`plot_utils.py`**: 封装了常用的`matplotlib`和`seaborn`绘图函数，如绘制混淆矩阵、ROC曲线、特征重要性等，实现了绘图逻辑与UI逻辑的分离。
*   **`feature_name_utils.py` & `prediction_utils.py`**: **关键模块**。实现了原始特征名与编码后特征名之间的映射。`FeatureNameMapper`类是核心，它在数据编码时记录映射关系，并在模型解释（SHAP）和手动预测时，将编码后的、对机器友好的特征名转换回人类可读的原始特征名。这极大地提升了应用的用户体验和结果的可解释性。
*   `enhanced_visualizations.py` & `shap_visualization_windows.py`: 提供高级或独立的特定可视化窗口，解决了`matplotlib`与`PyQt`在复杂场景下（如SHAP的某些原生绘图）的兼容性问题。

## 5. 技术实现与设计哲学

### 5.1. 异步处理与多线程
为了保证图形用户界面（GUI）的流畅性，所有耗时较长的计算任务（如特征选择、模型训练、优化算法）都被放在`QThread`子线程中执行。

*   **实现方式**: 每个需要异步处理的模块都有一个对应的`Worker`类（如`FeatureSelectionWorker`, `OptimizationWorker`）。主UI线程负责收集配置参数，然后实例化一个Worker并将其移动到新的`QThread`中。
*   **通信**: Worker与主UI线程之间通过`pyqtSignal`进行通信。例如，Worker会发射`progress_updated`信号来更新进度条，发射`finished`信号来传递最终结果，或发射`error_occurred`信号来报告错误。这种机制确保了所有UI更新都在主线程中安全地进行。

### 5.2. 数据流与信号/槽机制
模块之间的数据传递严格遵循信号/槽机制，实现了高度的解耦。

*   **数据流向**: `DataModule` -> `FeatureModule` -> `TrainingModule` -> (下游模块)。
*   **信号**: 每个上游模块在完成其任务后，会发射一个包含处理结果（如`pd.DataFrame`）的信号。
*   **槽**: 下游模块有一个或多个`pyqtSlot`修饰的槽函数，用于接收上游模块发射的信号并加载数据。
*   **优点**: 这种设计使得可以轻松地修改、添加或移除工作流中的某个模块，而不会影响到其他模块。例如，可以轻易地在`DataModule`和`FeatureModule`之间插入一个新的`AdvancedPreprocessing`模块。

### 5.3. 鲁棒性与错误处理
代码中包含了大量的`try...except`块，以确保单个组件的失败不会导致整个应用程序的崩溃。

*   **依赖检查**: 核心库（如`pymoo`, `shap`, `xgboost`）的导入都包裹在`try...except`中，如果导入失败，相关功能会被优雅地禁用，并向用户提供提示。
*   **用户输入验证**: 在执行计算密集型任务前，会对用户的输入（如文件路径、参数范围）进行严格验证。
*   **后备方案 (Fallback)**: 在许多关键功能中，如果首选的高级方法失败，系统会自动切换到一个更简单、更可靠的后备方案。例如，在SHAP分析中，如果`TreeExplainer`失败，系统会尝试使用`KernelExplainer`。

### 5.4. 特征名管理 (`feature_name_utils.py`)
这是提升用户体验和模型可解释性的一个关键设计。

*   **问题**: 机器学习预处理（特别是独热编码）会改变原始的特征名称，使得后续的分析结果（如特征重要性、SHAP值）难以理解。
*   **解决方案**:
    1.  在`DataModule`中进行编码时，通过`feature_name_mapper.record_encoding()`全局记录下原始特征名和编码后特征名之间的映射关系。
    2.  在需要向用户展示特征相关信息的地方（如`SHAPAnalysis`的图表、`PredictionModule`的手动输入界面），调用`get_readable_feature_names()`等函数将机器可读的编码名转换回人类可读的原始名（例如，`"材料类型_A"` -> `"材料类型: A"`）。
    3.  反之，在`PredictionModule`中，当用户输入原始特征值时，`map_original_to_encoded()`函数会根据记录的映射关系，将其转换为模型训练时使用的编码格式。

## 6. 未来展望与贡献
MatSci-ML Studio 作为一个强大的基础平台，未来还有广阔的扩展空间：

*   **模型库扩展**: 集成更多深度学习模型（如PyTorch, TensorFlow），特别是图神经网络（GNN），用于处理晶体结构等图形数据。
*   **数据库集成**: 直接连接Materials Project、AFLOW等材料数据库，实现在线数据获取。
*   **强化学习**: 引入强化学习算法，用于更智能的、序列化的实验设计。
*   **云端部署**: 将计算密集型任务（如HPO、GA）部署到云端服务器，解放本地计算资源。
*   **报告自动生成**: 进一步完善报告生成功能，可以一键导出包含所有分析步骤、结果和可视化的PDF或HTML报告。

## 7. 常见问题
*   ImportError: DLL load failed while importing pyarmor_runtime: 找不到指定的模块。
*   解决方案：电脑上缺少 Microsoft Visual C++ 运行库 (Microsoft Visual C++ Redistributable)解决方案：安装 Visual C++ 运行库
前往微软官方下载页面:https://learn.microsoft.com/zh-cn/cpp/windows/latest-supported-vc-redist?view=msvc-170
最新支持的 Visual C++ Redistributable 下载 | Microsoft Learn
选择正确的版本下载:
在页面上找到 "Visual Studio 2015, 2017, 2019 和 2022" 这一节。
根据您的 Python 解释器 是 64 位还是 32 位来选择下载。
如果您的系统是 64 位，并且 Python 也是 64 位（现在绝大多数情况都是），请下载 X64 版本。文件名为 VC_redist.x64.exe。
如果您的 Python 是 32 位，请下载 X86 版本。文件名为 VC_redist.x86.exe。
如果不确定，安装 X64 版本通常是正确的选择。
安装并重启:
下载后，双击运行 .exe 安装程序。
按照提示完成安装。如果系统提示您重启，请重启电脑以确保所有设置生效。
再次运行程序:
安装完成后，回到您的项目目录 (C:\Users\Rain\Desktop\MatSci-ML-Studio1-main\MatSci-ML-Studio1-main)，然后再次尝试运行 main.py。
python main.py
请确保使用的时 3.13+以上版本的python
欢迎使用者通过邮箱1255201958@qq.com反馈使用过程中的bug的所遇到的问题，我会在下面的版本迭代中解决这些问题
---


---

# **MatSci-ML Studio: An Intelligent Machine Learning Platform for Materials Science**

**Version**: 1.0
**Last Updated**: 2025.07.08

## ✨ Project Features

*   **Modular Workflow**: Clearly divides the process into multiple stages such as data management, feature engineering, model training, prediction, and optimization, aligning with the logical flow of scientific research.
*   **Graphical User Interface (GUI)**: All operations can be completed through an intuitive graphical interface, without the need to write complex code.
*   **Intelligent Wizard**: A built-in intelligent analysis and recommendation system that can automatically assess data quality and recommend suitable preprocessing, feature selection, and model configurations.
*   **Advanced Optimization Algorithms**: Integrates genetic algorithms, particle swarm optimization, Bayesian optimization, and multi-objective optimization (NSGA-II, MOEA/D, etc.) for inverse design and optimization of target properties.
*   **Model Interpretability**: Provides model explanation functionality based on SHAP (SHapley Additive exPlanations) to help users understand the model's decision-making process.
*   **Comprehensive Preprocessing**: Supports advanced outlier detection, smart missing value imputation, data transformation, and feature scaling.
*   **Collaboration & Version Control**: Built-in project management and version snapshot functionality to facilitate team collaboration and experiment traceability.
*   **Real-time Performance Monitoring**: Monitors system resource usage and task progress in real-time to ensure a smooth running experience.

---

## 📚 Module Breakdown

This project is composed of the following core modules, each of which can be found in a tab on the main interface:

1.  [**📊 Data Management**](#-module-1-data-management--preprocessing)
2.  [**🧬 Advanced Preprocessing**](#-module-2-advanced-preprocessing)
3.  [**🎯 Feature Selection**](#-module-3-feature-engineering--selection)
4.  [**🔬 Model Training**](#-module-4-model-training--evaluation)
5.  [**🧠 SHAP Analysis**](#-module-5-shap-model-interpretability-analysis)
6.  [**🎯 Target Optimization**](#-module-6-target-property-optimization)
7.  [**🔄 Multi-Objective Optimization**](#-module-7-multi-objective-optimization)
8.  [**🤖 Active Learning**](#-module-8-active-learning--optimization)
9.  [**💡 Intelligent Wizard**](#-module-9-intelligent-wizard)
10. [**⚙️ Performance Monitor**](#-module-10-performance-monitor)
11. [**🤝 Collaboration**](#-module-11-collaboration--version-control)
12. [**✅ Model Prediction**](#-module-12-model-prediction--results-export)

---

### 📊 Module 1: Data Management & Preprocessing

This is the starting point of the entire workflow, responsible for data import, exploration, cleaning, and initial preparation.

*   **Core Functions**:
    *   **Data Import**: Supports importing data from CSV, Excel files, and the system clipboard.
    *   **Data Exploration**: Provides a data overview (shape, types, missing values) and a table preview.
    *   **Data Visualization**: Automatically generates a data quality report, including a missing value heatmap, distribution plots, correlation matrix, etc., to help users intuitively understand the data.
    *   **Data Cleaning**: Provides functions for handling missing values (deletion, mean/median/mode imputation) and removing duplicate rows.
    *   **Feature/Target Selection**: Users can select feature columns (X) and a target column (y) through the interface.
    *   **Type Recommendation**: Automatically recommends the task type (classification or regression) based on the characteristics of the target column.
    *   **Categorical Feature Encoding**: Supports One-Hot Encoding, Label Encoding, etc., for categorical features.

*   **Core Class**: `DataModule(QWidget)`

---

### 🧬 Module 2: Advanced Preprocessing

This module provides a series of advanced data cleaning and transformation tools to improve data quality.

*   **Core Functions**:
    *   **Smart Recommendation System**: The built-in `SmartRecommendationEngine` can automatically analyze data quality and recommend suitable preprocessing steps.
    *   **Advanced Outlier Detection**: Integrates various outlier detection algorithms, such as `IQR`, `Z-Score`, `Isolation Forest`, `Local Outlier Factor`, etc., and provides visualization.
    *   **Smart Missing Value Imputation**: In addition to basic imputation methods, it also supports advanced imputation strategies such as `KNNImputer` and `IterativeImputer`.
    *   **Data Transformation**: Provides various data scaling and transformation tools like `StandardScaler`, `MinMaxScaler`, `PowerTransformer`, to meet the needs of different models.
    *   **State Management & History**: `StateManager` and `OperationHistory` implement powerful undo/redo functionality and a detailed operation log, ensuring all preprocessing steps are traceable.

*   **Core Classes**: `AdvancedPreprocessing(QWidget)`, `OutlierDetector`, `SmartImputer`, `StateManager`

---

### 🎯 Module 3: Feature Engineering & Selection

This module focuses on selecting the most valuable subset of features from the original set to improve model performance and reduce the risk of overfitting.

*   **Core Functions**:
    *   **Multi-Stage Selection Strategy**: Implements a multi-stage feature selection process of "Importance Filtering" -> "Correlation Filtering" -> "Advanced Search".
    *   **Model-Driven Importance Assessment**: Uses a machine learning model (e.g., Random Forest) to calculate feature importance scores and filters based on them.
    *   **Correlation Analysis**: Automatically detects and removes highly correlated redundant features, providing a before-and-after comparison of the correlation matrix.
    *   **Advanced Subset Search**:
        *   **Sequential Feature Selection (SFS)**: Supports forward and backward greedy search strategies.
        *   **Genetic Algorithm (GA)**: Searches for the optimal feature combination by simulating the process of biological evolution.
        *   **Exhaustive Search**: A brute-force search of all possible feature combinations (suitable for a small number of features).
    *   **Parallel Computing**: Utilizes `ThreadPoolExecutor` and `ProcessPoolExecutor` to accelerate computationally intensive feature evaluation processes.

*   **Core Classes**: `FeatureModule(QWidget)`, `FeatureSelectionWorker(QThread)`

---

### 🔬 Module 4: Model Training & Evaluation

This is the core of the machine learning workflow, responsible for training models with the processed data and comprehensively evaluating their performance.

*   **Core Functions**:
    *   **Rich Model Library**: Provides a comprehensive library of classification and regression models via `ml_utils`, including linear models, SVMs, tree-based models, ensemble models (Random Forest, Gradient Boosting), neural networks, etc.
    *   **Automated Preprocessing Pipeline**: Automatically constructs a `sklearn.pipeline.Pipeline` for numeric and categorical features, integrating missing value imputation, scaling, and encoding.
    *   **Hyperparameter Optimization (HPO)**: Supports various methods like Grid Search, Random Search, and Bayesian Search to automatically find the best parameters for a model.
    *   **Reliable Performance Evaluation**: Offers multiple evaluation strategies, including single split, multiple random splits, and nested cross-validation, to obtain an unbiased estimate of model performance.
    *   **Comprehensive Metrics & Visualization**: Automatically calculates and displays various metrics (e.g., R², MSE, Accuracy, F1-Score, ROC AUC) and generates various visualizations like confusion matrices, ROC curves, and residual plots.
    *   **Model Persistence**: Supports saving the entire trained pipeline (including the preprocessor and model) as a `.joblib` file for reuse in the prediction module.

*   **Core Class**: `TrainingModule(QWidget)`

---

### 🧠 Module 5: SHAP Model Interpretability Analysis

This module utilizes the SHAP (SHapley Additive exPlanations) framework to provide deep, intuitive explanations for trained "black-box" models.

*   **Core Functions**:
    *   **Automatic Explainer Selection**: Automatically selects the most appropriate SHAP explainer (`TreeExplainer`, `KernelExplainer`, etc.) based on the model type.
    *   **Global Explanations**:
        *   **Summary Plot / Beeswarm Plot**: Intuitively displays the overall impact direction and magnitude of each feature on the model's output.
        *   **Feature Importance**: Clearly displays the average importance ranking of features in a bar chart.
    *   **Local Explanations**:
        *   **Waterfall Plot**: Details the prediction for a single sample, showing how each feature pushes the prediction from a baseline value to the final value.
        *   **Force Plot**: Intuitively shows which features are the main forces pushing or pulling a single sample's prediction.
    *   **Feature Dependence Plots**: Reveals the effect of a single feature's marginal contribution on the model's output and can automatically discover and visualize interaction effects with other features.
    *   **Independent Visualization Windows**: Solves compatibility issues between Matplotlib and PyQt by opening all SHAP plots in separate, interactive windows for easy viewing and exporting.

*   **Core Classes**: `SHAPAnalysisModule(QWidget)`, `SHAPCalculationWorker(QThread)`

---

### 🎯 Module 6: Target Property Optimization

This is a powerful inverse design tool used to find the optimal combination of features that achieves a specific target material property.

*   **Core Functions**:
    *   **Multiple Optimization Algorithms**: Integrates Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and several advanced optimizers from `scipy` and `scikit-optimize` (e.g., Differential Evolution, Bayesian Optimization).
    *   **Constrained Optimization**: Supports user-defined boundary constraints and linear/non-linear constraints on feature variables.
    *   **Real-time Visualization**: During the optimization process, it plots convergence curves and parameter space exploration maps in real-time, allowing users to monitor the optimization process.
    *   **Thread-Safe**: All computationally intensive optimization tasks run in a separate `QThread`, ensuring a smooth and responsive GUI, with support for interruption.

*   **Core Classes**: `TargetOptimizationModule(QWidget)`, `OptimizationWorker(QThread)`

---

### 🔄 Module 7: Multi-Objective Optimization

When it's necessary to optimize multiple conflicting objectives simultaneously (e.g., maximizing both strength and toughness), this module can find a set of "best trade-off" solutions, known as the Pareto Front.

*   **Core Functions**:
    *   **Powerful Optimization Library**: Based on the `pymoo` library, it supports leading multi-objective optimization algorithms like `NSGA-II`, `NSGA-III`, `SPEA2`, and `MOEA/D`.
    *   **Mixed-Variable Handling**: Built-in custom sampling (`MixedVariableSampling`) and repair (`EnhancedBoundaryRepair`) operators provide native support for mixed feature types, including continuous, integer, binary, and categorical variables.
    *   **Robustness and Constraint Handling**: Offers advanced features like robust optimization (`RobustOptimizationHandler`) and explicit constraint handling (`ExplicitConstraintHandler`).
    *   **Real-time Monitoring & Visualization**: Uses a custom callback (`OptimizationCallback`) to update the Pareto front plot and convergence metrics (like hypervolume) in real-time.
    *   **Checkpoint Mechanism**: Supports automatically saving checkpoints (`OptimizationCheckpoint`) during the optimization process, allowing for resumption from an interruption.

*   **Core Classes**: `MultiObjectiveOptimizationModule(QWidget)`, `MLProblem(Problem)`

---

### 🤖 Module 8: Active Learning & Optimization

This module combines machine learning with experimental design, guiding the next "most valuable" experiment to run through an intelligent recommendation system, thus finding the optimal solution with the fewest experiments.

*   **Core Functions**:
    *   **Surrogate Model Library**: Integrates several powerful surrogate models (`RandomForest`, `XGBoost`, `GaussianProcess`, etc.) to learn and predict target properties.
    *   **Multiple Acquisition Functions**: Supports classic acquisition functions like Expected Improvement (EI) and Upper Confidence Bound (UCB) to balance exploration and exploitation.
    *   **Candidate Set Generation**: Provides various methods like Latin Hypercube Sampling, Sobol sequence, and Grid Search to generate the candidate experimental space.
    *   **Iterative Learning**: The entire module is designed around a closed loop of "recommend-feedback-relearn", with the `ActiveLearningSession` class managing the entire iterative process.
    *   **Comprehensive Process Visualization**: Plots learning curves, feature importance evolution, and exploration-exploitation balance in real-time to help users understand every step of the active learning process.

*   **Core Classes**: `ActiveLearningWindow(QMainWindow)`, `ActiveLearningOptimizer`

---

### 💡 Module 9: Intelligent Wizard

This module acts as the user's smart assistant, providing intelligent configuration recommendations for subsequent preprocessing and modeling steps through automated data analysis.

*   **Core Functions**:
    *   **Automated Data Analysis**: The `DataAnalysisWorker` thread performs a comprehensive statistical and quality assessment of the data in the background.
    *   **Multi-Dimensional Quality Assessment**: Evaluates data quality across multiple dimensions, including completeness, uniqueness, validity, consistency, and relevance.
    *   **Intelligent Recommendations**: Automatically recommends the best missing value handling methods, outlier detection strategies, feature selection techniques, and models based on the analysis results.
    *   **One-Click Apply**: Users can adopt the wizard's recommended configuration with a single click, which will then be automatically applied to the relevant modules.

*   **Core Classes**: `IntelligentWizard(QWidget)`, `DataAnalysisWorker(QThread)`

---

### ⚙️ Performance Monitor

A real-time monitoring tool for tracking the application's system resource usage and the execution progress of background tasks.

*   **Core Functions**:
    *   **System Resource Monitoring**: The `SystemMonitorWorker` uses the `psutil` library to monitor CPU, memory, disk, and network usage in real-time.
    *   **Task Progress Tracking**: The `TaskProgressTracker` can track the progress, duration, and status of long-running tasks (like feature selection and model training).
    *   **Real-time Charts**: Displays system performance data in real-time updating charts.
    *   **Performance Alerts**: Issues alerts when CPU or memory usage is too high, reminding the user of potential performance bottlenecks.

*   **Core Classes**: `PerformanceMonitor(QWidget)`, `SystemMonitorWorker(QThread)`

---

### 🤝 Collaboration & Version Control

This module provides basic support for team collaboration and experiment management.

*   **Core Functions**:
    *   **Project Management**: Supports creating, opening, and deleting projects, each with an independent directory structure for storing data, models, and results.
    *   **Version Snapshots**: The `VersionControl` class allows users to create a "snapshot" of the project's current state (including data and models) with a descriptive message.
    *   **History Tracking**: All version snapshots are recorded, allowing users to review and revert to previous experimental states at any time.
    *   **Project Export**: Supports packaging the entire project into a `.zip` file for easy sharing and migration.

*   **Core Classes**: `CollaborationWidget(QWidget)`, `ProjectManager`, `VersionControl`

---

### ✅ Module 12: Model Prediction & Results Export

This module is used to load a pre-trained model, make predictions on new data, and export the results.

*   **Core Functions**:
    *   **Model Loading**: Supports loading a model trained in the current session or a persisted model from a `.joblib` file.
    *   **Detailed Model Information**: Automatically parses and displays detailed metadata about the loaded model, including its type, parameters, feature names, and preprocessing steps.
    *   **Dual Prediction Modes**:
        *   **Batch Prediction**: Supports importing batch data from a file or clipboard for prediction.
        *   **Manual Input**: Automatically generates an input form based on the model's features, allowing users to input feature values for a single sample and get a real-time prediction.
    *   **Results Export**: Supports exporting the data with prediction results to a CSV or Excel file.

*   **Core Class**: `PredictionModule(QWidget)`


---

## 🚀 Recommended Workflow

To fully leverage MatSci-ML Studio, we recommend the following workflow. This process generally aligns with the order of the module tabs in the main interface, designed to guide you seamlessly from data to insight.

1.  **Data Import & Exploration** (in the `Data Management` module)
    *   **Starting Point**: Begin here. Import your raw dataset using the `Browse` or `From Clipboard` buttons.
    *   **Explore**: In the right-hand panel, carefully review the "Data Overview" and "Data Quality" tabs to understand your data's basic information, preview its content, and check for missing values and distributions.
    *   **Initial Cleaning**: Use the "Data Cleaning" section to handle obvious missing values and duplicate data.
    *   **Define Task**: In the "Feature & Target Selection" section, choose your features (X) and target (y). The system will automatically recommend a task type (classification/regression).
    *   **Encoding (Optional)**: If your data contains non-numeric categorical features, use the "Categorical Feature Encoding" function to encode them.

2.  **In-depth Data Processing** (in `Advanced Preprocessing` or `Intelligent Wizard` modules)
    *   **Intelligent Wizard**: If you're unsure how to process your data, switch to the `Intelligent Wizard` module. Run the intelligent analysis, and it will provide a complete set of recommendations for preprocessing and modeling.
    *   **Manual Advanced Processing**: Alternatively, in the `Advanced Preprocessing` module, you can manually perform more advanced operations, such as detecting and handling outliers with various sophisticated algorithms.

3.  **Feature Selection** (in the `Feature Selection` module)
    *   **Dimensionality Reduction**: Load the preprocessed data into this module. By configuring a multi-stage feature selection strategy (e.g., importance filtering, correlation filtering, and advanced search), you can eliminate redundant or irrelevant features to improve model performance.

4.  **Model Training** (in the `Model Training` module)
    *   **Core Step**: Use the selected feature set to train machine learning models.
    *   **Select Model**: Choose one or more candidate models based on your task type.
    *   **Hyperparameter Optimization (Optional)**: Enable HPO (Hyperparameter Optimization) to automatically find the best parameter combination for your model.
    *   **Evaluate**: After training is complete, comprehensively evaluate the model's performance in the right-hand panel's "Metrics," "Confusion Matrix," and other tabs.
    *   **Save Model**: Once you are satisfied with the results, click `Save Model` to save the entire trained pipeline.

5.  **Model Application & Exploration** (parallel use of multiple modules)
    *   Once you have a trained model, you can proceed to any of the following modules based on your needs:
    *   **`Prediction`**: Load the model to predict properties for new, unknown data.
    *   **`SHAP Analysis`**: Load the model and data to gain deep insights into *why* the model makes its predictions and analyze feature influences.
    *   **`Target Optimization` / `Multi-Objective Optimization`**: If your goal is "inverse design" (i.e., finding compositions or parameters that produce specific properties), use these modules. Load your model(s), set your optimization targets and constraints, and the algorithms will search for the optimal feature combinations.
    *   **`Active Learning`**: If your goal is to "guide the next experiment," use this module. It will recommend the most informative "next experiment point" based on the current model, helping you achieve your goal with the fewest experiments.

6.  **Project Management & Monitoring** (throughout the process)
    *   **`Collaboration`**: At any key stage of your project, you can use this module to create a "snapshot" to version your work for team sharing and reproducibility.
    *   **`Performance Monitor`**: When performing any computationally intensive task (like training or optimization), you can switch to this module at any time to check system resource usage.

---

### 🛠️ Core Utilities & Helper Modules

This section is primarily for developers who want to understand the project's underlying logic or contribute to its development.

*   **`main_window.py`**
    *   **Role**: The application's main entry point and framework.
    *   **Responsibility**: Creates the main window, initializes and contains all functional module tabs (`QTabWidget`), and establishes the signal-slot connections (`connect_modules`) between them, ensuring a smooth flow of data and state between different modules.

*   **`ml_utils.py`**
    *   **Role**: The "core brain" of machine learning.
    *   **Responsibility**: Encapsulates all underlying machine learning logic. It defines the platform's supported `sklearn` model library (`get_available_models`), provides default hyperparameter ranges (`get_default_hyperparameters`), and contains core functions for model evaluation and preprocessing pipeline creation.

*   **`data_utils.py` & `plot_utils.py`**
    *   **Role**: Foundational toolkits for data and visualization.
    *   **Responsibility**: `data_utils.py` encapsulates all data import and basic quality check functionalities. `plot_utils.py` provides a series of standardized plotting functions to ensure a consistent and reusable visualization style throughout the application.

*   **`feature_name_utils.py` & `prediction_utils.py`**
    *   **Role**: Feature name mapping and management system.
    *   **Responsibility**: This is a critical component for handling complex features, especially after one-hot encoding. The `FeatureNameMapper` class is responsible for tracking the mapping between original features and their encoded names. This allows modules like `SHAPAnalysis` and `PredictionModule` to continue using human-readable original feature names, greatly enhancing user experience and the interpretability of results.

*   **`shap_visualization_windows.py`**
    *   **Role**: An independent rendering solution for SHAP visualizations.
    *   **Responsibility**: To resolve compatibility issues between `matplotlib` and `PyQt` in certain environments (especially in background threads), this file defines independent `QDialog` windows. When a user needs to view a SHAP plot, the program creates a new, separate window to render and display the image, ensuring the stability and responsiveness of the main interface.

*   **`enhanced_visualizations.py`** (found in `active_learning_optimizer.py`)
    *   **Role**: A dedicated visualization toolkit for the Active Learning module.
    *   **Responsibility**: While not a top-level module, it provides the `ActiveLearningWindow` with a variety of advanced visualizations, such as PCA dimensionality reduction analysis, uncertainty analysis, design space slicing, and optimization history, which are key to providing deep insights in the active learning module.

## 🚀 Getting Started

### Install Dependencies

```bash
pip install PyQt5 numpy pandas scikit-learn matplotlib seaborn joblib scipy openpyxl xlrd
```

### Launch the Application

```bash
python main.py
```
*Note: Depending on your project structure, the entry point might be `main.py`.*

## 🔧 System Requirements

- Python 3.12+
- PyQt5 5.15.0+
- Scientific Computing Libraries: NumPy, Pandas, SciPy
- Machine Learning Libraries: scikit-learn, XGBoost, LightGBM, CatBoost
- Visualization Libraries: Matplotlib, Seaborn
- Model Interpretation Library: SHAP

## 📞 Contact

For questions or suggestions, please contact: 1255201958@qq.com

---

## 6. Technical Implementation & Philosophy

-   **Asynchronous Processing**: All time-consuming tasks are executed in `QThread`s, communicating with the main UI thread via signals and slots to ensure a responsive interface.
-   **Decoupled Data Flow**: Modules are decoupled and pass data through signals, making the workflow flexible and easy to extend.
-   **Robustness & Error Handling**: Extensive `try...except` blocks and fallback mechanisms ensure application stability.
-   **Feature Name Management**: A global feature name mapping system solves the common problem of feature names changing after encoding, dramatically improving usability and interpretability.

## 7. Future Work & Contribution

-   **Model Expansion**: Integrate deep learning models, especially Graph Neural Networks (GNNs).
-   **Database Integration**: Connect to materials databases like Materials Project.
-   **Reinforcement Learning**: Introduce RL for more intelligent, sequential experimental design.
-   **Cloud Deployment**: Offload heavy computations to cloud servers.

Contributions are welcome! Please feel free to fork the repository, submit pull requests, or open issues to suggest improvements. Or contact me at 1255201958@qq.com.



