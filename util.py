import pandas as pd
import random

class QuestionBank:
    def __init__(self, csv_file_path):
        # Load the CSV file and extract the 'Question' column
        self.df = pd.read_csv(csv_file_path)
        
        # Ensure the 'Question' column exists in the CSV file
        if 'paraphrased' not in self.df.columns:
            raise ValueError("CSV file must contain a 'Question' column")
        
        # Store the list of questions
        self.questions = self.df['paraphrased'].tolist()

    def get_question(self, n):
        """
        Randomly sample n questions from the question list.
        
        Parameters:
        - n (int): The number of questions to sample.
        
        Returns:
        - list: A list of n randomly sampled questions.
        """
        if n > len(self.questions):
            raise ValueError("Requested sample size exceeds the number of available questions.")
        
        return random.sample(self.questions, n) 

# # Usage example:
# # Initialize the question bank with the path to your CSV file
# question_bank = QuestionBank('path_to_your_csv_file.csv')

# # Get a sample of 5 questions
# sample_questions = question_bank.get_question(5)

# # Print the sampled questions
# print("Sampled Questions:")
# for question in sample_questions:
#     print(question)
