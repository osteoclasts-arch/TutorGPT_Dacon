import pandas as pd
import numpy as np
import os
from typing import Tuple

class MockDataGenerator:
    """
    Generates mock data for the Dacon Learner Completion Prediction competition.
    Simulates the structure of survey data with various features.
    """

    def __init__(self, output_dir: str = "data", random_state: int = 42):
        self.output_dir = output_dir
        self.random_state = random_state
        np.random.seed(random_state)
        os.makedirs(output_dir, exist_ok=True)

    def generate_data(self, n_samples: int = 1000, is_train: bool = True) -> pd.DataFrame:
        """Generates a dataframe with mock student data."""
        
        # Categorical choices
        majors = ['Engineering', 'Arts', 'Business', 'Science', 'Other']
        inflow_routes = ['Social Media', 'Search Engine', 'Recommendation', 'Advertisement', 'Email']
        genders = ['Male', 'Female']
        education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
        
        data = {
            'ID': [f'{"TRAIN" if is_train else "TEST"}_{i:05d}' for i in range(n_samples)],
            'generation': np.random.choice(range(20, 70), n_samples),
            'major_data': np.random.choice(majors, n_samples),
            'inflow_route': np.random.choice(inflow_routes, n_samples),
            'gender': np.random.choice(genders, n_samples),
            'education_level': np.random.choice(education_levels, n_samples),
            'previous_courses_cnt': np.random.poisson(lam=2, size=n_samples),
            'learning_hours_avg': np.random.normal(loc=10, scale=3, size=n_samples),
            'quiz_score_avg': np.random.uniform(50, 100, n_samples),
            'platform_usage_days': np.random.randint(0, 30, n_samples),
            'survey_satisfaction': np.random.randint(1, 6, n_samples),
            # Add some missing values
            'employment_status': np.random.choice(['Employed', 'Unemployed', np.nan], n_samples, p=[0.7, 0.2, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        if is_train:
            # Generate target with some correlation to features
            # e.g., higher quiz score and more learning hours -> higher chance of completion
            prob = (
                (df['quiz_score_avg'] - 50) / 50 * 0.4 + 
                (df['learning_hours_avg'] / 20) * 0.3 + 
                np.random.normal(0, 0.1, n_samples)
            )
            # Normalize prob to 0-1
            prob = (prob - prob.min()) / (prob.max() - prob.min())
            df['target'] = (prob > 0.5).astype(int)
            
        return df

    def save_csv(self):
        """Generates and saves train and test datasets."""
        print(f"Generating mock data in {self.output_dir}...")
        
        train_df = self.generate_data(n_samples=2000, is_train=True)
        test_df = self.generate_data(n_samples=1000, is_train=False)
        
        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Saved train.csv ({len(train_df)} rows) and test.csv ({len(test_df)} rows).")

if __name__ == "__main__":
    generator = MockDataGenerator(output_dir="/home/junsikback/.openclaw/workspace/TutorGPT_Dacon/data")
    generator.save_csv()
