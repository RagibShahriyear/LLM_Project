# WIP


import json
import os
import csv
from extract import extract_general_info, extract_treatments, predict_treatments
from similarity import compute_similarity_score
from graph import save_similarity_score, plot_similarity_scores, read_csv

current_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_dir, "data", "similarity_scores.csv")

# load a note
notes = ["Note 1: Discharge Summary: Patient: 60-year-old male with moderate ARDS from COVID-19 Hospital Course: The patient was admitted to the hospital with symptoms of fever, dry cough, and dyspnea. During physical therapy on the acute ward, the patient experienced coughing attacks that induced oxygen desaturation and dyspnea with any change of position or deep breathing. To avoid rapid deterioration and respiratory failure, a step-by-step approach was used for position changes. The breathing exercises were adapted to avoid prolonged coughing and oxygen desaturation, and with close monitoring, the patient managed to perform strength and walking exercises at a low level. Exercise progression was low initially but increased daily until hospital discharge to a rehabilitation clinic on day 10. Clinical Outcome: The patient was discharged on day 10 to a rehabilitation clinic making satisfactory progress with all symptoms resolved. Follow-up: The patient will receive follow-up care at the rehabilitation clinic, with regular monitoring of progress and further rehabilitation exercises until full recovery. Any new symptoms or concerns should be reported to the clinic immediately. Overall Impression: The patient responded well to treatment, and with appropriate medical intervention, was able to overcome the difficulties faced during hospitalization for ARDS from COVID-19. The patient's level of care was of a high standard, with all necessary therapy provided and monitoring of progress before discharge." , "Note 2: Discharge Summary: Admission Date: [Insert Date] Discharge Date: [Insert Date] Patient Name: [Insert Name] Age/Sex: 39-year-old Male Medical Record Number: [Insert Number] Admission Diagnosis: Unspecified respiratory tract infection Discharge Diagnosis: Resolved respiratory tract infection Hospital Course: The patient was hospitalized due to persistent fever and dry cough for 2 weeks, leading to reduced general health condition. On admission, the patient required 4 L/min of oxygen due to rapid shallow breathing at rest and severe breathlessness during minor physical activity. The initial physical therapy focused on educating the patient on dyspnea-relieving positions, mobilization techniques, and deep-breathing exercises. However, with time, the patient's anxiety-induced dyspnea became an issue, leading to the modification of therapy to relieve his dyspnea. The patient positively responded to the therapy, evidenced by a reduction in respiratory rate from 30 to 22 breaths/min and an increase in oxygen saturation from 92% to 96% on 4 L/min oxygen. Over the next few days, his dyspnea and anxiety started to alleviate gradually, and the patient regained his self-confidence. The therapy was eventually shifted to walking and strength training, and the patient was able to walk for 350m without a walking aid or supplemental oxygen before discharge. Hospital Outcome: The patient was discharged home in an improved condition, with resolved respiratory tract infection, showing no need for further hospitalization. The patient was advised to follow-up with his regular healthcare professional for further monitoring and management of his health status. Discharge Medications: None Follow-up: The patient was advised to follow-up with his regular healthcare professional regarding further monitoring and management of his health status. Impression: Resolved respiratory tract infection with successful physical therapy for dyspnea relief and anxiety reduction. The patient responded well to therapy and regained his self-confidence, able to walk independently before discharge.", "Note 3: Hospital Course Summary: Admission Date: [Insert date] Discharge Date: [Insert date] Patient: [Patient's Name] Sex: Male Age: 57 years Admission Diagnosis: Oxygen Desaturation Hospital Course: The patient was admitted to the ICU one week after a positive COVID-19 result due to oxygen desaturation. Physical therapy was initiated promptly after admission, which helped improve the patient's breathing frequency and oxygen saturation. The patient was guided to achieve a prone position resulting in a significant increase in oxygen saturation from 88% to 96%. The patient continued to receive intensive physical therapy, positioning, and oxygen therapy for the next few days. Although there were challenges in achieving the prone position due to the patient's profoundly reduced respiratory capacity and high risk of symptom exacerbation, the medical team succeeded in implementing a safe and individualized approach. After three days with this regime, the patient was transferred to the normal ward, where physical therapists continued his rehabilitation, including walking and strength training. However, the patient's severe instability remained a challenge. Nevertheless, after nine days from ICU admission, the patient was successfully discharged from the hospital as a pedestrian. Discharge Condition: At the time of discharge, the patient's medical condition had significantly improved, and he was considered stable enough to be discharged from the hospital. The patient's oxygen saturation had returned to normal limits, and his breathing frequency had decreased significantly. Summary: This course summary demonstrates that the patient responded positively to a physical therapy treatment regimen, including positioning, deep-breathing exercises, and walking. Although the patient's medical condition was quite severe during the initial ICU admission, his rehabilitation resulted in marked improvement, leading to a successful discharge from the hospital.", "Note 4: Discharge Summary: Patient: 69-year-old male Hospital Course: The patient was admitted to the ICU due to COVID-19 pneumonia, which was diagnosed after he experienced a dry cough for 2 weeks and showed poor oxygenation. Initially, he was given lung-protective ventilation and targeted sedation, and he remained stable. However, his condition worsened over the next days, and he developed hemodynamic instability and severe Acute Respiratory Distress Syndrome (ARDS). The patient underwent intermittent prone positioning and continuous renal replacement therapy because of these complications. Physical therapists were involved in this process, and they were responsible for ensuring the correct positioning of the patient's joints and preventing secondary complications, such as pressure ulcers, nerve lesions and contractures. After the tracheostomy, the patient underwent passive range-of-motion exercises and passive side-edge mobilization. However, asynchronous ventilation and hemodynamic instability persisted, inhibiting active participation. After spending 24 days in the ICU, the patient showed severe signs of muscle loss and scored 1/50 points on the Chelsea Critical Care Physical Assessment Tool (CPAx). The patient died soon after the withdrawal of life support. Hospital Stay: The patient was in the ICU for 24 days, where he received treatment interventions, including passive range of motion and positioning, intermittent prone positioning, and continuous renal replacement therapy. While in the ICU, the patient showed severe signs of muscle loss and developed pressure ulcer on the forehead. Discharge Instruction: The patient has been discharged in a stable condition. Although the patient was not discharged, we offer our sincere condolences to his family.", "Note 5: Discharge Summary: Patient Information: - Name: [Redacted] - Age: 57 - Gender: Male - Admission Date: [Redacted] - Discharge Date: N/A - Medical Record Number: [Redacted] Hospital Course Summary: - The patient was admitted to the ICU with symptoms of dyspnea, heavy dry cough, and fever after testing positive for COVID-19. - Despite initial ability to exercise and sit in a chair with a physical therapist, the patient's respiratory condition progressively worsened, requiring intubation and proning. - The patient's large amounts of bronchial mucus and respiratory failure necessitated regular suctioning and respiratory therapy. - Manual airway clearance techniques were employed by 1-2 physical therapists to increase effective airway clearance while avoiding alveolar collapse with some success. - After extubation, the patient continued to require intensive manual airway clearance techniques, nasal rinsing, and airway suctioning. - Additional physical therapy interventions, including passive range of motion, assisted exercising, and mobilization, were also utilized. - At the time of writing, the patient remained in the ICU without ventilatory support. Summary of Events: - The patient presented with symptoms of COVID-19, including dyspnea, heavy dry cough, and fever. - Despite intubation and proning, the patient's respiratory condition continued to worsen, requiring regular suctioning and respiratory therapy. - Manual airway clearance techniques were used to increase expiratory flow for effective airway clearance. - After extubation, the patient continued to require intensive manual airway clearance techniques, nasal rinsing, and airway suctioning. - The patient also received additional physical therapy interventions to aid in recovery. - At the time of writing, the patient remained in the ICU without ventilatory support. Impression: - The patient presented with symptoms of COVID-19 and suffered from respiratory failure necessitating intubation, proning, manual airway clearance techniques, and extensive respiratory therapy. - Despite initial improvements, the patient continues to require intensive manual airway clearance techniques, nasal rinsing, airway suctioning, and additional physical therapy. - The patient has not yet been discharged."]


# extract
note = notes[4]
general_info = extract_general_info(note)
treatments_info = extract_treatments(note)
predicted_treatments = predict_treatments(general_info)
    


# similarity 
similarity_score = compute_similarity_score(treatments_info, predict_treatments)

# save score
patient_id = "note 5"
filename = output_file
save_similarity_score(patient_id, similarity_score, filename)

# Read the CSV file
read_csv(filename)

# Plot the data
plot_similarity_scores(patient_id, similarity_score)