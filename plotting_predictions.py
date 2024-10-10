import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import PRED_DIR, PLOT_DIR


def extract_accuracies_from_file(file_path):
    accuracies = []
    file_name = os.path.basename(file_path)

    with open(file_path, 'r') as file:
        content = file.read()

        # Extract the temperatures and corresponding average accuracies
        temp_accuracy_pattern = r'Average metrics for Temperature (\d+):.*?Average Accuracy: ([\d\.]+)'
        matches = re.findall(temp_accuracy_pattern, content, re.DOTALL)

        for match in matches:
            temperature, accuracy = match
            step = re.search(r'step(\d+)', file_name).group(1)  # Extract step from filename

            # Adjusting for model size extraction based on "small" or "middle"
            if "small" in file_name:
                model_size = "0.5"
            elif "middle" in file_name:
                model_size = "1.5"
            else:
                model_size = "unknown"  # Default if no pattern matches

            accuracies.append({
                'file_name': file_name,
                'step': step,
                'model_size': model_size,
                'temperature': int(temperature),
                'accuracy': float(accuracy)
            })

    return accuracies



def process_files_in_directory(directory_path):
    all_accuracies = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            accuracies = extract_accuracies_from_file(file_path)
            all_accuracies.extend(accuracies)
    print(all_accuracies)
    return all_accuracies


def plot_accuracies(accuracies, plot_dir):
    df = pd.DataFrame(accuracies)

    # Convert step and temperature to categorical for better plot
    df['temperature'] = df['temperature'].astype(str)

    # Loop over unique steps and model sizes
    for step in df['step'].unique():
        step_data = df[df['step'] == step]

        plt.figure(figsize=(10, 6))

        # Create barplot for each step, showing model size in different colors
        sns.barplot(x='temperature', y='accuracy', hue='model_size', data=step_data, palette='Set2')

        plt.title(f'Accuracy for Different Temperatures - Step {step}')
        plt.xlabel('Temperature')
        plt.ylabel('Average Accuracy')
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.legend(title='Model Size')

        # Save the plot to the plot directory
        plot_file_path = os.path.join(plot_dir, f'step_{step}_accuracy_plot.png')
        plt.savefig(plot_file_path)
        plt.close()


# Step 4: Execute the process
directory_path = PRED_DIR
plots_directory = PLOT_DIR
accuracies_data = process_files_in_directory(directory_path)

# Plot and save the results in the plot directory
plot_accuracies(accuracies_data, plots_directory)
