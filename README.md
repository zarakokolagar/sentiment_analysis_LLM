# Task: Sentiment Analysis with IMDB Dataset 

## Description:

### Task Objectives

1. Design and refine prompts to achieve maximum accuracy in sentiment analysis
2. Implement local inference for SLMs using Python
3. Compare the performance and efficiency of two SLMs of two sizes
4. Analyze results and draw meaningful insights



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

Alternatively, you could run the notebook on [Colab Notebook](https://colab.research.google.com/drive/1UWiUKyRz0HgGtB_frLV6Kl0MMb7J25Kg?usp=sharing)
If you are running the code from Google colab, all required libraries and data handling is included in the colab code.



#### Acknowledgement of Limitations:

1. **Sample Size:** Due to computational constraints, I was only able to test the models on 1000 samples (500 per class). While this provides an initial indication of model performance, predictions may fluctuate when applied to larger datasets. I addressed this to some extent by using a more diverse data sampling strategy to ensure broader coverage of examples.


2. **Execution Environment:** The model inference was conducted on Google Colab, which is linked in my GitHub repository via the README.md file. The code is available in both the README and the Jupyter notebook in the repository. Colab’s limitations, including personal account restrictions and potential connection issues, led me to save the prediction results after each run to ensure stability. Plotting is also based on these saved predictions, as connection issues in Colab could cause cell outputs to be lost. Please refer to the plots directory for the visualized results.


3. **Parameter Tuning:** I focused exclusively on experimenting with different temperature settings and did not explore the influence of parameters like top-p and top-k. This decision was made for several reasons:

    a. **Simplicity and Clarity:** By controlling only the temperature, I ensured that the results are easier to interpret. Introducing multiple varying parameters could make it harder to attribute performance changes to any one factor.
    
    b. **Task Relevance:** The temperature parameter directly impacts the randomness and creativity of model outputs, which is closely related to sentiment prediction tasks. In contrast, top-p and top-k are more suited for tasks requiring diverse content generation.
    
    c. **Resource Efficiency:** Given the focus on smaller language models, reducing the parameter space helped minimize the computational load and experimentation time.
    
    d.**Comparability:** Keeping the scope to temperature adjustments allowed for a clearer comparison between models of different sizes without adding complexity from other variables.
    
    
4. **Prompting Approach:** Since the step 4 prompting approach yielded stable and satisfactory results for both models, I opted to stop further prompt experimentation. However, I acknowledge that additional prompting techniques could be explored to ensure robustness in real-world applications. Further tests such as cross-validation across different data domains, error analysis of edge cases, and A/B testing with alternative prompt designs would be necessary to refine the model's reliability for production-level sentiment analysis.




