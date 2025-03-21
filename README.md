# Nectar - Negative-Correlation-based TCM Architecture for Reversal

## Project Description / 项目描述

**English:**  
Nectar is a Python package designed to optimize Traditional Chinese Medicine (TCM) herbal formulas using data-driven techniques. It processes input data (herbal information and disease data) and optimizes herb ratios to generate a formulation with minimized score. The package includes modules for data preprocessing, herb filtering, dosage–weight conversion, optimization, and visualization.

**中文：**  
Nectar 是一个利用数据驱动方法优化中药方剂的 Python 包。它处理中药信息和疾病数据，并通过优化药材比例生成一个评分最小化的方剂。该包包含数据预处理、药材筛选、剂量与权重转换、优化计算和结果可视化等模块。

---

## Installation / 安装

### Requirements / 依赖
- Python 3.8 or higher / Python 3.8 或更高版本  
- Required packages: numpy, pandas, torch, matplotlib, tqdm, scikit-learn, scipy, seaborn, dill, openpyxl  
  所需依赖：numpy, pandas, torch, matplotlib, tqdm, scikit-learn, scipy, seaborn, dill, openpyxl

### Installation Steps / 安装步骤

1. **Clone the repository / 克隆仓库:**
   ```bash
   git clone <repository_url>
   cd nectar
   ```

2. **Create and activate a virtual environment (recommended) / 创建并激活虚拟环境（推荐）:**
   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux/MacOS
   venv\Scripts\activate        # Windows
   ```

3. **Install the package / 安装包:**
   ```bash
   pip install .
   ```
   *Or install dependencies from requirements.txt:*
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage / 用法

### Command-Line Interface (CLI) / 命令行界面

After installation, run the main optimization pipeline by executing the `nectar` command.  
安装后，通过执行 `nectar` 命令启动主要的优化流程（由 `main.py` 定义）：

```bash
nectar --herb_info_path path/to/info_input_herbs.xlsx --disease_data_path path/to/disease_nes.pkl
```

- If no arguments are provided, default file paths within the code will be used.  
  若未提供参数，则使用代码中设定的默认路径。

The CLI will output the optimized herbal formula and score, and save detailed results (including plots) in a timestamped results folder.  
命令行界面会输出优化后的方剂及得分，并将详细结果（包括图表）保存到以时间戳命名的文件夹中。

### Library Usage / 库调用

You can also use Nectar as a library within your own Python scripts:

```python
from nectar.main import nectar  # Import the main pipeline function

# Run the optimization pipeline with custom file paths
result = nectar("path/to/info_input_herbs.xlsx", "path/to/disease_nes.pkl")

print("Optimized formula:", result["final_formula"])
print("Final score:", result["final_score"])
```

The returned `result` is a dictionary with:
- `final_formula`: Optimized list of herbs  
- `dosage`: Corresponding dosages  
- `final_score`: Optimization score  
- `result_folder`: Directory containing detailed results and plots  

返回的 `result` 是一个字典，包含：
- `final_formula`: 优化后的药材列表  
- `dosage`: 对应剂量  
- `final_score`: 优化得分  
- `result_folder`: 保存详细结果和图表的文件夹路径

---

## Project Structure / 项目结构

```
nectar/         # Project root (项目根目录)
├── nectar/     # Python package directory (包目录)
│   ├── __init__.py
│   ├── main.py         # Main pipeline and CLI entry point (主流程及命令行入口)
│   └── modules/        # Core modules (核心模块)
│       ├── __init__.py
│       ├── herb_filter.py             # Herb filtering module (药材筛选模块)
│       ├── herb_ratio_optimization.py # Herb ratio optimization (药材比例优化)
│       ├── plotting.py                # Visualization module (结果可视化)
│       ├── seed_utils.py              # Random seed utilities (随机种子工具)
│       ├── weight_to_dosage.py        # Weight-to-dosage conversion (权重转剂量)
│       ├── calculateScore.py          # Scoring function module (评分计算模块)
│       ├── data_io.py                 # Data I/O functions (数据加载模块)
│       ├── data_preprocessing.py      # Data preprocessing (数据预处理)
│       └── dosage_to_weight.py        # Dosage-to-weight conversion (剂量转权重)
├── data/              # (Optional) Data files for testing (测试数据文件)
├── README.md          # Project documentation (本文件)
├── pyproject.toml     # Build configuration and metadata (构建配置)
├── requirements.txt   # Dependency list (依赖列表)
└── .gitignore         # Git ignore rules (Git忽略规则)
```

---

## Author / 作者
**Zheng Wu**
