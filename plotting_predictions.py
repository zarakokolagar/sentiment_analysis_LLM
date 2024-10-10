import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import PRED_DIR, PLOT_DIR
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AccuracyData:
    file_name: str
    step: str
    model_size: str
    temperature: int
    accuracy: float


def extract_accuracies_from_file(file_path: str) -> List[AccuracyData]:
    """Extract accuracy data from a single file."""
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

            # Determine model size from the filename
            model_size = determine_model_size(file_name)

            # Append extracted data as an AccuracyData object
            accuracies.append(AccuracyData(
                file_name=file_name,
                step=step,
                model_size=model_size,
                temperature=int(temperature),
                accuracy=float(accuracy)
            ))

    return accuracies


def determine_model_size(file_name: str) -> str:
    """Determine the model size based on the filename."""
    if "small" in file_name:
        return "0.5"
    elif "middle" in file_name:
        return "1.5"
    else:
        return "unknown"  # Default if no pattern matches


def process_files_in_directory(directory_path: str) -> List[AccuracyData]:
    """Process all text files in a given directory and return a list of AccuracyData."""
    all_accuracies = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            accuracies = extract_accuracies_from_file(file_path)
            all_accuracies.extend(accuracies)

    return all_accuracies


def plot_accuracies(accuracies: List[AccuracyData], plot_dir: str) -> None:
    """Plot accuracies and save plots to the specified directory."""
    df = pd.DataFrame([data.__dict__ for data in accuracies])

    # Convert step and temperature to categorical for better plotting
    df['temperature'] = df['temperature'].astype(str)

    # Loop over unique steps and model sizes to create plots
    for step in df['step'].unique():
        step_data = df[df['step'] == step]

        plt.figure(figsize=(10, 6))
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


def main():
    """Main function to process accuracy files and generate plots."""
    directory_path = PRED_DIR
    plots_directory = PLOT_DIR

    accuracies_data = process_files_in_directory(directory_path)

    # Plot and save the results in the plot directory
    plot_accuracies(accuracies_data, plots_directory)


if __name__ == "__main__":
    main()
