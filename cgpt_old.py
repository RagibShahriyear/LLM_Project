from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json

# Load environment variables from .env
load_dotenv()

# Initialize the language model
llm = ChatGroq(model="mixtral-8x7b-32768")

# Function to extract general information
def extract_general_info(note):
    template = """
    Given the following medical note, extract the following information and format it as JSON:
    - visit motivation
    - admission
    - patient information
    - patient medical history
    - surgeries
    - symptoms
    - medical examination
    - diagnostic tests
    - discharge

    Note: Do not include 'treatments' in this extraction.

    Medical Note:
    {note}

    Extracted Information:
    """
    prompt = PromptTemplate(template=template, input_variables=["note"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(note=note)
    return json.loads(response)

# Function to extract treatments
def extract_treatments(note):
    template = """
    Given the following medical note, extract the 'treatments' information and format it as JSON:

    Medical Note:
    {note}

    Extracted Treatments:
    """
    prompt = PromptTemplate(template=template, input_variables=["note"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(note=note)
    return json.loads(response)

# Example usage
medical_note = """
A 47-year-old male patient was referred to the rheumatology clinic because of recurrent attacks of pain in both knees over 1 year.
In September 2016, the patient presented with severe pain over the medial aspect of the left knee for a two-week duration which prevented him from ambulation. The pain increased with weight-bearing physical activity. The patient reported no history of trauma before the onset of the knee pain. Examination showed severe tenderness over the medial side of the knee with mild effusion and moderate limitation of range of motion. There was no erythema or increased warmth of the knee. MRI of the left knee showed a moderate-sized focal area of marrow edema/contusion involving the medial femoral condyle in mid and anterior parts predominantly along the articular surface. The patient was prescribed diclofenac sodium 50 mg twice daily and was advised to avoid prolonged weight-bearing activities. Over the next few weeks, the pain subsided and resolved. Three months later, the patient developed spontaneous new onset of pain involving the lateral aspect of the same knee. MRI showed bone marrow edema involving the lateral femoral condyle with complete resolution of the bone marrow edema of the medial femoral condyle. He was treated conservatively with NSAIDs and physiotherapy and advised to use cane to minimize weight bearing on the diseased knee. demonstrates MRI of the left knee in September 2016 and three months later.
In April 2017, the patient developed gradual pain over the medial side of the right knee with no obvious swelling. MRI of the right knee showed a moderate-sized focal area of marrow edema involving the medial tibial plateau medially and anteriorly. The patient was treated conservatively in a similar fashion to the previous episode. Four months later, the pain got more severe for which he underwent another MRI of the right knee which showed extensive marrow edema involving the medial femoral condyle with complete recovery of the medial tibial plateau bone marrow edema noted in the previous MRI (). The patient also recalled a similar pain happened in 2011 to the left knee but did not do MRI at that time.
In all previous presentations, the patient did not report any history of trauma, fall, twist, constitutional symptoms, or using corticosteroids. He also had no history of other joint involvement apart from knees and denied any history of low back pain. He did not have any features suggestive of spondyloarthropathy or connective tissue disease.
Past history is significant for fracture of the greater tuberosity of the left humerus and undisplaced fracture of the left cuboid bone. Fractures happened after he fell off a ladder. Also, he is known to have mild asthma which is controlled with as-needed bronchodilator and hypertension maintained on amlodipine 5 mg daily. The patient had never been a smoker or an alcohol consumer.
Lab investigations revealed vitamin D 8 ng/mL (normal: >30 ng/mL), corrected calcium 2.16 mmol/L (normal: 2.10–2.60 mmol/L), parathyroid hormone 91 pg/ml (normal: 15–65 pg/ml), and alkaline phosphatase 49 U/L (normal: 40–150 U/L). Complete blood count, kidney and liver function, CRP, and ESR were within normal limit. Immunology profile including rheumatoid factor, ACPA, ANA, anticardiolipin, and B2-glycoprotein were all negative.
DXA scan showed a T score of −1.0 at the lumbar spine and −1.6 at the left femoral neck suggestive of osteopenia. shows further details of the DXA scan.
The patient was treated conservatively with oral vitamin D2 50,000 IU/week supplement and NSAIDs. Gradually, the symptoms subsided over the next few weeks, and vitamin D level became normal after 12 weeks.
"""

general_info = extract_general_info(medical_note)
treatments_info = extract_treatments(medical_note)

print("General Information:\n", json.dumps(general_info, indent=4))
print("\nTreatments Information:\n", json.dumps(treatments_info, indent=4))
