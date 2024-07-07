# Save the score 
import os 
import csv
from similarity import *

current_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_dir, "data", "similarity_scores.csv")



def save_similarity_score(patient_id, similarity_score, filename):
    # Check if the file already exists
    file_exists = False
    try:
        with open(filename, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass

    # Open the CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file does not already exist
        if not file_exists:
            writer.writerow(["Patient ID", "Similarity Score"])
        
        # Write the patient ID and similarity score to the file
        writer.writerow([patient_id, similarity_score])

# Example usage
patient_id = "note 2"
similarity_score = similarity_score
filename = output_file
save_similarity_score(patient_id, similarity_score, output_file)



# Plot the score 

import csv
import matplotlib.pyplot as plt

# Read the CSV file
def read_csv(filename):
    patient_ids = []
    similarity_scores = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            patient_ids.append(row[0])
            similarity_scores.append(float(row[1]))
    return patient_ids, similarity_scores

# Plot the data
def plot_similarity_scores(patient_ids, similarity_scores):
    plt.figure(figsize=(10, 5))
    plt.scatter(patient_ids, similarity_scores, color='blue')
    plt.xlabel('Patient ID')
    plt.ylabel('Similarity Score')
    plt.title('Similarity Scores by Patient ID')
    plt.ylim(0, 10)
    plt.grid(axis='y')
    plt.show()

# Example usage
filename = 'data/similarity_scores.csv'
patient_ids, similarity_scores = read_csv(filename)
plot_similarity_scores(patient_ids, similarity_scores)


