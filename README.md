# Nectar - Negative-Correlation-based TCM Architecture for Reversal

## Project Description

Nectar is a Python package designed to optimize Traditional Chinese Medicine (TCM) herbal formulas using data-driven techniques. It processes input data (herbal information and disease data) and optimizes herb ratios to generate a formulation with minimized score. The package includes modules for data preprocessing, herb filtering, dosage–weight conversion, optimization, and visualization.

---

## Installation

### Requirements
- Python 3.8 or higher / Python 3.8 或更高版本  
- Required packages: numpy, pandas, torch, matplotlib, tqdm, scikit-learn, scipy, seaborn, dill, openpyxl  

### Installation Steps / 安装步骤

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChenHuangMUST/NeCTAR.git
   cd nectar
   ```

2. **Create and activate a virtual environment (recommended):**

    - **On Linux/MacOS:**
      ```bash
      python -m venv venv
      source venv/bin/activate
      ```
    - **On Windows:**
      ```bash
      python -m venv venv
      venv\Scripts\activate
      ```

3. **Install dependencies:**
   *Install dependencies from requirements.txt:*
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage / 用法

### Command-Line Interface (CLI) / 命令行界面

After installation, run the main optimization pipeline by executing the `nectar` command.  

```bash
nectar --herb_info_path path/to/info_input_herbs.xlsx --disease_data_path path/to/disease_nes.pkl
```

- If no arguments are provided, default file paths within the code will be used.  

The CLI will output the optimized herbal formula and score, and save detailed results (including plots) in a timestamped results folder.  

### Library Usage

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

---

## Author / 作者
**Zheng Wu**
