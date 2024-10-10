# Task: Sentiment Analysis with IMDB Dataset 

## Description:

### Task Objectives

1. Design and refine prompts to achieve maximum accuracy in sentiment analysis
2. Implement local inference for SLMs using Python
3. Compare the performance and efficiency of two SLMs of different sizes
4. Analyze results and draw meaningful insights
5. Document the process and findings effectively


### Data

The dataset of IMDB movie reviews is accessed through [Huggingface](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews), which contains around 50k samples; 40k for 'train' and 10k for 'test' dataset.

### Models 

1. [bartowski/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/blob/main/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf): A lightweight 1.5B-parameter instruction-
tuned model with enhanced capabilities in coding, mathematics, and handling structured data.


2. [bartowski/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/blob/main/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf): A model with 500M parameters, making it
three times smaller than its 1.5B counterpart, yet still highly capable for many tasks.


### Project Structure

The project is organized as follows:

    sentiment_analysis/
    ├── config.py
    ├── data_loader.py
    ├── data/
    │   ├── sampled_dataset.csv
    ├── predictions/
    │   ├── this folder contains text file of all predictions from model inference
    ├── plots/
    │   ├─ this folder contains plots for each prompting step
    ├── Qwen_sentiment_analysis.ipynb
    ├── requirements.txt
    ├── README.md

1. config.py: Contains configurations such as directory paths and dataset locations.

2. data_loader.py : Contains code for dataset sampling.

3. data/: Directory containing the dataset sampled and used in the project.

4. predictions/: contains 8 text files consisting of model predictions for each step.

5. plots/: contains 4 plots for average model accuracies across 3 runs per temperature [0,1,2] for each model based on the 4 prompting steps.

6. Qwen_sentiment_analysis.ipynb: The Jupyter notebook containing the analysis and model-building code.  


### Installation

To run this project, follow the steps below:

1. Clone the Project: Download the project as a zip file or clone it.

2. Set Up the Environment: Install the required Python packages using pip:

```python
pip install -r requirements.txt
```

### Usage

### Setting Up a Virtual Environment and Installing Dependencies

To run this project in a local Jupyter Notebook, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zarakokolagar/sentiment_analysis_LLM
   cd sentiment_analysis_LLM  

## Setting Up a Virtual Environment and Running the Project

Follow these steps to set up a virtual environment and run the Jupyter Notebook locally:

#### 1. Install `virtualenv` (if not already installed)

If you don’t have `virtualenv` installed, you can install it using:

```bash
pip install virtualenv
```

#### 2. Create a virtual environment
Create a virtual environment by running the following command:
```bash
virtualenv venv
```
#### 3. Activate the virtual environment
On macOS/Linux:

```bash
source venv/bin/activate
```
#### 4. Install the required dependencies
With the virtual environment activated, install the dependencies listed in the `requirements.txt` file

```bash
pip install -r requirements.txt
```

#### 5. Run Jupyter Notebook
If you don’t have Jupyter Notebook installed, you can install it with:
```bash
pip install notebook
```
Then, start Jupyter Notebook by running:

```bash
jupyter notebook
```

#### 6. Deactivate the virtual environment
Once you're done working, deactivate the virtual environment by running:

```bash
deactivate
```

**Note:**

Alternatively, you could run the notebook on [Colab Notebook](https://colab.research.google.com/drive/1UWiUKyRz0HgGtB_frLV6Kl0MMb7J25Kg#scrollTo=ihomXxB4Svsk&uniqifier=1)
If you are running the code from Google colab, all required libraries and data handling is included in the colab code.



## License 

Not required