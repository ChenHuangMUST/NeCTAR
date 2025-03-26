# NeCTAR - Negative-Correlation-based TCM Architecture for Reversal

## Project Description

Nectar is a Python package designed to optimize Traditional Chinese Medicine (TCM) herbal formulas using data-driven techniques. It processes input data (herbal information and disease data) and optimizes herb ratios to generate a formulation with minimized score. The package includes modules for data preprocessing, herb filtering, dosage–weight conversion, optimization, and visualization.

---

## Installation

### Requirements
- Python 3.8 or higher
- Required packages: numpy, pandas, torch, matplotlib, tqdm, scikit-learn, scipy, seaborn, dill, openpyxl  

### Installation Steps

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

## Usage

Enter the project directory and install
Open a command line tool, navigate to the root directory of the NeCTAR project, and then execute:
```bash
pip install -e .
```

### Command-Line Interface (CLI)

After installation, run the main optimization pipeline by executing the `nectar` command.  

```bash
cd NeCTAR
python nectar/main.py --herb_info_path path/to/info_input_herbs.xlsx --disease_data_path path/to/disease_nes.pkl
```
- info_input_herbs is an initial formula dataset with two columns. The first column is "name", where each row represents the Chinese name of a traditional Chinese medicine herb, and the second column is "dosage", a decimal that indicates the dosage of the herb. If an herb not used during training is entered, the program will return a lookup failure error.
- disease_nes is the GSEA result for the disease and must include the columns ['ID', 'NES'], where "ID" is the pathway ID and "NES" is the normalized enrichment score. In fact, the program can handle disease_nes provided either in the .pkl or .txt format, as long as the file contains the required columns.
- If no arguments are provided, default premature ovarian failure file paths within the code will be used. 

The CLI will output the optimized herbal formula and score, and save detailed results (including plots) in a timestamped results folder.  

### Library Usage

You can also use Nectar as a library within your own Python scripts:

Usage in Your Python Script
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

## Author
**Zheng Wu**
