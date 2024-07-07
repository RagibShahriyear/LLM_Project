from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from extract import treatments_info, predicted_treatments
import re

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatGroq(model="mixtral-8x7b-32768")

similarity_template = """Given the following two medical treatment plans, compare their contents, look into their meanings, and provide a similarity score between 0 and 10 (where 0 means the two treatment plans are completely different, and 10 means they are exactly the same). The score should be a integer, no decimal points. 

Medical Treatment Plan 1:
{treatment_plan_1}

Medical Treatment Plan 2:
{treatment_plan_2}

Similarity Score:

"""

def strip_non_numeric(string):
    # Use a regular expression to remove all characters except digits and the decimal point
    return re.sub(r'[^0-9.]', '', string)

def compute_similarity_score(treatment_info, predicted_treatments):
    similarity = ChatPromptTemplate.from_template(similarity_template)
    similarity_prompt = similarity.invoke({f"treatment_plan_1": treatments_info, "treatment_plan_2": predicted_treatments})
    similarity_analysis = model.invoke(similarity_prompt)
    detailed_score = similarity_analysis.content
    final_score = model.invoke(f"Extract the score from this: {detailed_score}. The output should be the final score, as an int, no other word. Example: '6', etc.")
    similarity_score = int(strip_non_numeric(final_score.content))
    return (similarity_score)


similarity_score = compute_similarity_score(treatment_info = treatments_info, predicted_treatments = predicted_treatments)
print(similarity_score)




# --------------------------------------------------------------------------------------------------

# def strip_non_numeric(string):
#     return re.sub(r'[^0-9.]', '', string)

# def compute_similarity_score(treatment_plan_1, treatment_plan_2):
#     # Create a ChatPromptTemplate from the similarity template
#     similarity = ChatPromptTemplate.from_template(similarity_template)
    
#     # Generate the prompt
#     prompt = similarity.invoke({"treatment_plan_1": treatment_plan_1, "treatment_plan_2": treatment_plan_2})
    
#     # Get the detailed score from the model
#     result = model.invoke(prompt)
#     detailed_score = result.content
    
#     # Extract the final score from the detailed output
#     final_score = model.invoke(f"Extract the score from this: {detailed_score}. The output should be the final score, as a float, no other word. Example: '0.555', etc.")
    
#     # Function to strip everything except numbers and decimal point from a string
    
#     # Convert the extracted score to float
#     similarity_score = float(strip_non_numeric(final_score.content))
    
#     return similarity_score

# compute_similarity_score(treatments_info, predicted_treatments)
