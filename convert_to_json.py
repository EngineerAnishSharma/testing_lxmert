import pandas as pd
import json
from sklearn.model_selection import train_test_split

def convert_to_lxmert_json(dataframe, output_json_path):
    """
    Converts a Pandas DataFrame with preprocessed VQA data into the JSON format
    expected by the Lxmert repository.

    Args:
        dataframe: The Pandas DataFrame containing the preprocessed data.
        output_json_path: The path to save the output JSON file.
    """
    formatted_questions = []

    for index, row in dataframe.iterrows():
        # Extract the image ID from the formatted_image_id
        image_id = int(row['formatted_image_id'].split('_')[-1].split('.')[0])

        # Convert the comma-separated answers string to a list
        answers_list = [answer.strip() for answer in row['answers'].split(',')]

        formatted_questions.append({
            "question": row['neutral_question'],
            "image_id": image_id,
            "question_id": row['question_id'],
            "answers": answers_list,
            "image_path": row['formatted_image_id'],  # Assuming you want to keep this field
        })

    # Create the final JSON structure
    output_json = {
        "info": {
            "description": "Preprocessed VQA dataset with bias mitigation",
            "data_source": "mscoco",
            "split": "train"  # or "val" or "test" as appropriate
        },
        "questions": formatted_questions
    }

    # Save the JSON to a file
    with open(output_json_path, 'w') as f:
        json.dump(output_json, f, indent=4)

# Example Usage:
# Assuming your DataFrame is named 'df_preprocessed'
df_preprocessed = pd.read_csv("preprocessed_data.csv")  # Load your data

# Split into train and val sets (80% train, 20% val)
df_train, df_val = train_test_split(df_preprocessed, test_size=0.2, random_state=42)

# Create the training JSON file
output_train_json_path = "data/vqa_bias_mitigated/preprocessed_vqa_bias_mitigated_train.json"
convert_to_lxmert_json(df_train, output_train_json_path)

# Create the validation JSON file
output_val_json_path = "data/vqa_bias_mitigated/preprocessed_vqa_bias_mitigated_val.json"
convert_to_lxmert_json(df_val, output_val_json_path)