# Prepare Ollama Docker image
# docker pull ollama/ollama
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# docker exec -it ollama ollama run llama3.2

# Install required libraries: pip install pandas pyspark ollama tqdm
# PySpark requires Java, install latest JDK from: https://www.openlogic.com/openjdk-downloads
# Don't forget to let the JDK installer set JAVA_HOME environment variable

# Import required libraries
import json
import re
import pandas as pd
from   pyspark.sql import SparkSession
from   ollama import Client
from   tqdm import tqdm

# Set Pandas display options
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

# Define Ollama API endpoint
API_URL = 'http://localhost:11434'

# Read in sample JSON file
filename = 'yelp_academic_dataset_review.json'

# Sample random records using PySpark
def spark_sample(filename, samples=1000):
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("RandomSampling") \
        .getOrCreate()
    df_spark = spark.read.json(filename)
    # Get random samples. `fraction` is the proportion of data to sample
    # Note: This method won't sample the exact number of records specified
    fraction = samples / df_spark.count()
    # Perform the sampling
    sampled_df = df_spark.sample(fraction=fraction, withReplacement=False)
    # Convert the sampled Spark DataFrame to a Pandas DataFrame
    df = sampled_df.toPandas()
    # Delete unused variables
    del filename, df_spark, fraction, sampled_df
    # Return Pandas dataframe
    return df

# Define prompt template for LLM
prompt_template =  """
Analyze the following user review and concisely extract what the user liked and disliked in the following format:
Liked: [item1, item2, ...]
Disliked: [item1, item2, ...]
If there are no relevant items in either Liked or Disliked, say "None"
Only respond with the list of Liked and Disliked, do not output anything else.
Review: {content}
"""

# Run LLM with given prompt and parameters
def run_llm(messages, api_url=API_URL, context_length=4096, num_predict=100, repeat_penalty=1.2):
    # (Optional) Define the system message to set the assistant's role and instructions
    system_message = {
        'role': 'system',
        'content': 'You are an expert AI research assistant, specialized with extracting useful and actionable insights from research papers.'
    }
    client = Client(host=api_url)
    response = client.chat(
        model='llama3.2',
        messages=messages,
        options={
            'num_ctx': context_length,
            'num_predict': num_predict,
            'repeat_penalty': repeat_penalty,
        }
    )
    return response['message']['content']

# Format input to LLM's expected format and run
def get_llm_output(text):
    user_message = {
        'role': 'user',
        'content': prompt_template.format(content=text)
    }
    messages = [user_message]
    llm_output = run_llm(messages).strip()
    return llm_output

# Parse LLM output to extract Liked and Disliked
def parse_llm_output(text):
    # Extract Liked
    pattern1 = r"Liked:\s*(.+?)\n"
    match1 = re.search(pattern1, text)
    if match1:
        # Extract the list of items, split by comma, and strip any whitespace
        items1 = [item.strip() for item in match1.group(1).split(',')]
    else:
        items1 = None

    # Extract Disliked
    pattern2 = r"Disliked:\s*(.+)"
    match2 = re.search(pattern2, text)
    if match2:
        # Extract the list of items, split by comma, and strip any whitespace
        items2 = [item.strip() for item in match2.group(1).split(',')]
    else:
        items2 = None
    # Return Liked and Disliked as tuple
    return (items1, items2)

# Overall function to take a Series from iteration, run LLM, parse and return in JSON
def extract_likes_dislikes(record, debug=False):
    review_id = record['review_id']
    business_id = record['business_id']
    user_id = record['user_id']
    text = record['text']
    if debug:
        print(f"Input (Biz ID {business_id}, Review ID {review_id}):\n{text}")
    llm_output = get_llm_output(text)
    if debug:
        print(f"Output:\n{llm_output}")
    liked, disliked = parse_llm_output(llm_output)
    review_record = {
        "review_id": review_id,
        "business_id": business_id,
        "user_id": user_id,
        "text": text,
        "liked": liked,
        "disliked": disliked,
    }
    return review_record

# Extract random samples from data
df = spark_sample(filename, samples=50)
# Initialize list
reviews = []
# Iterate through samples and extract Liked/Disliked
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    reviews.append(extract_likes_dislikes(row))

# View output as a DataFrame
pd.DataFrame(reviews)[['review_id', 'liked', 'disliked']]
