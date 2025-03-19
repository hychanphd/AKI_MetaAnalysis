import pandas as pd


class code2name:
    
    def __init__(self):
        pass
        
    def load_concept(self):
        path_concept ="/blue/yonghui.wu/hoyinchan/concept_vocab/"
        concept = pd.read_csv(path_concept+'CONCEPT.csv', sep='\t')
        concept['vocabulary_id'] = concept['vocabulary_id'].replace('CPT4', 'HCPCS')
        concept['concept_code'] = concept['concept_code'].astype(str)
        self.concept = concept
        
    def translate_omop(self, label, class_label = None, extra = None):
        def get_vocabulary_id(prefix):
            return {
                'LAB': 'LOINC',
                'MEDRX': 'RxNorm',
                'PXCH': 'HCPCS',
                'PX09': 'ICD9Proc',
                'PX10': 'ICD10PCS',
                'MEDATC':'ATC'
            }.get(prefix, None)

        try:
            if class_label is None:
                prefix, code, extra = label.split(':')
                try:
                    extra = extra.split('(')[0]    
                except:
                    pass
            #    extra = extra.replace('.','')
        
                if prefix == 'PX':
                    prefix = prefix+code
                if prefix == 'MED':
                    prefix = prefix+code  
                if prefix == 'DX':
                    prefix = prefix+code 
            else:
                prefix = class_label
                extra = extra
                
            label_omop = self.concept[(self.concept['concept_code'] == extra) & (self.concept['vocabulary_id'] == get_vocabulary_id(prefix))]['concept_name'].iloc[0]
        except:
            label_omop = label
        return label_omop
    
    #omop_label = {x:translate_omop(x, concept) for x in df_importances_stat[df_importances_stat['Label_rank']!=-100].index}
    # first_dixt = {x:plotshapsn.translate_omop(x) for x in plotshapsn.shapdf['Feature'].unique()}
    # Then ask ChatGPT to extract "extract the medical procedure, medication or Lab measurement from..."
    
    def custom_translate_omop(self, label):
        extracted_data = {
            'AGE': 'AGE',
            'DX:09:428bt6': 'Heart failure (428)',
            'LAB::10466-1': 'Anion gap 3 (10466-1)',
            'LAB::14979-9': 'aPTT (14979-9)',
            'LAB::17861-6': 'Calcium (17861-6)',
            'LAB::1863-0': 'Anion gap 4 (1863-0)',
            'LAB::19123-9': 'Magnesium (19123-9)',
            'LAB::1920-8': 'Aspartate aminotransferase (1920-8)',
            'LAB::1962-0': 'Deprecated Bicarbonate (1962-0)',
            'LAB::1963-8': 'Bicarbonate (1963-8)',
            'LAB::2028-9': 'Carbon dioxide, total (2028-9)',
            'LAB::20570-8': 'Hematocrit (20570-8)',
            'LAB::2075-0': 'Chloride (2075-0)',
            'LAB::2160-0': 'Creatinine (2160-0)',
            'LAB::2340-8': 'Glucose (2340-8)',
            'LAB::2345-7': 'Glucose (2345-7)',
            'LAB::26464-8': 'Leukocytes (26464-8)',
            'LAB::26478-8': 'Lymphocytes/100 leukocytes (26478-8)',
            'LAB::2708-6': 'Oxygen saturation (2708-6)',
            'LAB::3094-0': 'Urea nitrogen (3094-0)',
            'LAB::3097-3': 'Urea nitrogen/Creatinine ratio (3097-3)',
            'LAB::33037-3': 'Anion gap (33037-3)',
            'LAB::38483-4': 'Creatinine (38483-4)',
            'LAB::4092-3': 'Vancomycin (4092-3)',
            'LAB::41653-7': 'Glucose (41653-7)',
            'LAB::43413-4': 'Blood product units requested (43413-4)',
            'LAB::4544-3': 'Hematocrit (4544-3)',
            'LAB::48642-3': 'GFR predicted among non-blacks (48642-3)',
            'LAB::5902-2': 'Prothrombin time (PT) (5902-2)',
            'LAB::6690-2': 'Leukocytes (6690-2)',
            'LAB::713-8': 'Eosinophils/100 leukocytes (713-8)',
            'LAB::731-0': 'Lymphocytes (731-0)',
            'LAB::736-9': 'Lymphocytes/100 leukocytes (736-9)',
            'LAB::777-3': 'Platelets (777-3)',
            'LAB::788-0': 'Erythrocyte distribution width (788-0)',
            'LAB::789-8': 'Erythrocytes (789-8)',
            'LAB::LG5665-7': 'Alkaline phosphatase (LG5665-7)',
            'MED:ATC:C03CA': 'Sulfonamides, plain (C03CA)',
            'MED:ATC:J01CG': 'Beta-lactamase inhibitors (J01CG)',
            'ORIGINAL_BMI': 'BMI',
            'PX:09:39.61': 'Extracorporeal circulation auxiliary to open heart surgery (39.61)',
            'PX:09:39.95': 'Hemodialysis (39.95)',
            'PX:09:96.72': 'Continuous invasive mechanical ventilation >96 hrs (96.72)',
            'PX:09:99.04': 'Transfusion of packed cells (99.04)',
            'PX:10:5A1955Z': 'Respiratory Ventilation >96 hrs (5A1955Z)',
            'PX:CH:36415': 'Venous blood collection (36415)',
            'PX:CH:97116': 'Therapeutic procedure; gait training (97116)',
            'PX:CH:A6257': 'Transparent film dressing (A6257)',
            'PX:CH:J1940': 'Furosemide injection (J1940)',
            'PX:CH:J2543': 'Piperacillin/tazobactam injection (J2543)',
            'LAB::2777-1': 'Phosphate (789-8)',
            'LAB::1975-2': 'Bilirubin.total (1975-2)',
            'LAB::2823-3': 'Potassium (2823-2)',
            'LAB::2951-2': 'Sodium (2951-2)',
            'LAB::718-7': 'Hemoglobin (718-7)'
        }
        return extracted_data.get(label, label)

# Instruction for CHATGPT
# extract the medical procedure, medication or Lab measurement from the dict, return a python dict, Only include the most important concept
# Remove theose stuff luike "Number Concentration", "n Serum or Plasm", "by clculation" etc    
# Put the concept code in bracket at the end of the names
# make LAB:: to LAB:LONIC: inside the name, do not touch the key
    
    def custom_translate_omop_2022(self, label):
        extracted_data = {
         'LAB::10466-1': 'Anion gap 3 (LAB:LONIC:10466-1)',
         'LAB::18182-6': 'Osmolality of Serum (LAB:LONIC:18182-6)',
         'LAB::19023-1': 'Granulocytes/100 leukocytes (LAB:LONIC:19023-1)',
         'LAB::1962-0': 'Deprecated Bicarbonate in Plasma (LAB:LONIC:1962-0)',
         'LAB::20570-8': 'Hematocrit (LAB:LONIC:20570-8)',
         'LAB::2703-7': 'Oxygen (LAB:LONIC:2703-7)',
         'LAB::3173-2': 'aPTT (LAB:LONIC:3173-2)',
         'LAB::32623-1': 'Platelet mean volume (LAB:LONIC:32623-1)',
         'LAB::4544-3': 'Hematocrit (LAB:LONIC:4544-3)',
         'LAB::48642-3': 'Glomerular filtration rate (LAB:LONIC:48642-3)',
         'LAB::48643-1': 'Glomerular filtration rate (LAB:LONIC:48643-1)',
         'LAB::5902-2': 'Prothrombin time (PT) (LAB:LONIC:5902-2)',
         'LAB::5905-5': 'Monocytes/100 leukocytes (LAB:LONIC:5905-5)',
         'LAB::713-8': 'Eosinophils/100 leukocytes (LAB:LONIC:713-8)',
         'LAB::736-9': 'Lymphocytes/100 leukocytes (LAB:LONIC:736-9)',
         'LAB::770-8': 'Neutrophils/100 leukocytes (LAB:LONIC:770-8)',
         'LAB::788-0': 'Erythrocyte distribution width (LAB:LONIC:788-0)',
         'LAB::LG12083-8': 'Urea nitrogen/Creatinine (LAB:LONIC:LG12083-8)',
         'LAB::LG13614-9': 'Anion gap (LAB:LONIC:LG13614-9)',
         'LAB::LG32846-4': 'Granulocytes (LAB:LONIC:LG32846-4)',
         'LAB::LG32857-1': 'Leukocytes (LAB:LONIC:LG32857-1)',
         'LAB::LG32885-2': 'Monocytes (LAB:LONIC:LG32885-2)',
         'LAB::LG32892-8': 'Platelets (LAB:LONIC:LG32892-8)',
         'LAB::LG4454-7': 'Carbon dioxide (LAB:LONIC:LG4454-7)',
         'LAB::LG49829-1': 'Albumin (LAB:LONIC:LG49829-1)',
         'LAB::LG49936-4': 'Potassium (LAB:LONIC:LG49936-4)',
         'LAB::LG50477-5': 'Vancomycin (LAB:LONIC:LG50477-5)',
         'LAB::LG6037-8': 'Base deficit (LAB:LONIC:LG6037-8)',
         'LAB::LG6139-2': 'Anion gap 4 (LAB:LONIC:LG6139-2)',
         'LAB::LG6373-7': 'Chloride (LAB:LONIC:LG6373-7)',
         'LAB::LG6657-3': 'Creatinine (LAB:LONIC:LG6657-3)',
         'LAB::LG7967-5': 'Glucose (LAB:LONIC:LG7967-5)',
         'MED:ATC:H02AB': 'Glucocorticoids (MED:ATC:H02AB)',
         'MED:ATC:N02BE': 'Anilides (MED:ATC:N02BE)',
         'PX:CH:36415': 'Collection of venous blood by venipuncture (PX:CH:36415)',
         'PX:CH:94002': 'Ventilation assist and management (PX:CH:94002)',
         'PX:CH:97116': 'Therapeutic procedure, gait training (PX:CH:97116)',
         'PX:CH:97535': 'Self-care/home management training (PX:CH:97535)',
         'PX:CH:A6257': 'Transparent film, sterile (PX:CH:A6257)',
         'PX:CH:J1940': 'Injection, furosemide (PX:CH:J1940)',
         'PX:CH:J2543': 'Injection, piperacillin sodium/tazobactam sodium (PX:CH:J2543)',
         'PX:CH:J8499': 'Prescription drug, oral, non chemotherapeutic (PX:CH:J8499)',
         'LAB::LG50024-5': 'Creatinine (LAB:LOINC:LG50024-5)'
            
                }
        
        return extracted_data.get(label, label)
    
    def custom_translate_omop_2022_2(self, label):
        extracted_data = {
            'LAB::LG6657-3': 'Creatinine (LAB:LONIC:LG6657-3)',
            'LAB::48642-3': 'Glomerular filtration rate predicted among non-blacks (LAB:LONIC:48642-3)',
            'LAB::LG32857-1': 'Leukocytes (LAB:LONIC:LG32857-1)',
            'SYSTOLIC': 'SYSTOLIC',
            'DIASTOLIC': 'DIASTOLIC',
            'PX:CH:36415': 'Collection of venous blood by venipuncture (PX:CH:36415)',
            'LAB::LG6373-7': 'Chloride (LAB:LONIC:LG6373-7)',
            'LAB::LG13614-9': 'Anion gap (LAB:LONIC:LG13614-9)',
            'LAB::48643-1': 'Glomerular filtration rate predicted among blacks (LAB:LONIC:48643-1)',
            'LAB::LG2807-8': 'Bicarbonate (LAB:LONIC:LG2807-8)',
            'LAB::LG49864-8': 'Calcium (LAB:LONIC:LG49864-8)',
            'LAB::LG49936-4': 'Potassium (LAB:LONIC:LG49936-4)',
            'LAB::LG7967-5': 'Glucose (LAB:LONIC:LG7967-5)',
            'LAB::LG4454-7': 'Carbon dioxide (LAB:LONIC:LG4454-7)',
            'LAB::10466-1': 'Anion gap 3 (LAB:LONIC:10466-1)',
            'MED:ATC:B05XA': 'Electrolyte solutions (MED:ATC:B05XA)',
            'LAB::LG1314-6': 'Urea nitrogen (LAB:LONIC:LG1314-6)',
            'AGE': 'AGE',
            'LAB::18182-6': 'Osmolality (LAB:LONIC:18182-6)',
            'ORIGINAL_BMI': 'BMI',
            'LAB::LG49883-8': 'Glucose (LAB:LONIC:LG49883-8)',
            'LAB::LG32892-8': 'Platelets (LAB:LONIC:LG32892-8)',
            'PX:CH:97530': 'Therapeutic activities (PX:CH:97530)',
            'LAB::LG49949-7': 'Phosphate (LAB:LONIC:LG49949-7)',
            'LAB::713-8': 'Eosinophils (LAB:LONIC:713-8)',
            'LAB::LG11363-5': 'Sodium (LAB:LONIC:LG11363-5)',
            'LAB::736-9': 'Lymphocytes (LAB:LONIC:736-9)',
            'PX:CH:97116': 'Gait training (PX:CH:97116)',
            'LAB::LG50041-9': 'Magnesium (LAB:LONIC:LG50041-9)',
            'LAB::1962-0': 'Deprecated Bicarbonate (LAB:LONIC:1962-0)',
            'PX:CH:A6257': 'Transparent film, sterile (PX:CH:A6257)',
            'LAB::LG6139-2': 'Anion gap 4 (LAB:LONIC:LG6139-2)',
            'LAB::2703-7': 'Oxygen partial pressure (LAB:LONIC:2703-7)',
            'LAB::LG6037-8': 'Base deficit (LAB:LONIC:LG6037-8)',
            'LAB::LG344-8': 'Carbon dioxide (LAB:LONIC:LG344-8)',
            'LAB::LG12083-8': 'Urea nitrogen/Creatinine ratio (LAB:LONIC:LG12083-8)',
            'SEX_F': 'SEX_F',
            'PX:CH:36556': 'Insertion of central venous catheter (PX:CH:36556)',
            'PX:CH:J8499': 'Prescription drug, oral, non-chemotherapeutic (PX:CH:J8499)',
            'PX:CH:97535': 'Self-care/home management training (PX:CH:97535)',
            'MED:ATC:N02BE': 'Anilides (MED:ATC:N02BE)',
            'PX:CH:J3490': 'Unclassified drugs (PX:CH:J3490)',
            'LAB::786-4': 'MCHC (LAB:LONIC:786-4)',
            'LAB::26478-8': 'Lymphocytes (LAB:LONIC:26478-8)',
            'MED:ATC:A06AA': 'Softeners, emollients (MED:ATC:A06AA)',
            'MED:ATC:L03AC': 'Interleukins (MED:ATC:L03AC)',
            'LAB::LG50477-5': 'Vancomycin trough (LAB:LONIC:LG50477-5)',
            'LAB::LG32850-6': 'Erythrocytes (LAB:LONIC:LG32850-6)',
            'LAB::5905-5': 'Monocytes (LAB:LONIC:5905-5)',
            'LAB::LG44868-4': 'Hemoglobin (LAB:LONIC:LG44868-4)',
            'LAB::LG32885-2': 'Monocytes (LAB:LONIC:LG32885-2)',
            'LAB::788-0': 'Erythrocyte distribution width (LAB:LONIC:788-0)',
            'PX:CH:J3370': 'Injection, vancomycin hcl (PX:CH:J3370)',
            'LAB::LG5665-7': 'Alkaline phosphatase (LAB:LONIC:LG5665-7)',
            'PX:CH:96374': 'Therapeutic, prophylactic, or diagnostic injection (PX:CH:96374)',
            'MED:ATC:A07AA': 'Antibiotics (MED:ATC:A07AA)',
            'PX:CH:99024': 'Postoperative follow-up visit (PX:CH:99024)',
            'LAB::19023-1': 'Granulocytes (LAB:LONIC:19023-1)',
            'LAB::3173-2': 'aPTT (LAB:LONIC:3173-2)',
            'PX:CH:J2543': 'Injection, piperacillin sodium/tazobactam sodium (PX:CH:J2543)',
            'MED:RX:1719287': 'furosemide injection (MED:RX:1719287)',
            'LAB::30385-9': 'Erythrocyte distribution width (LAB:LONIC:30385-9)',
            'DX:09:584bt6': 'Acute kidney failure (DX:09:584bt6)',
            'LAB::4544-3': 'Hematocrit (LAB:LONIC:4544-3)',
            'LAB::14979-9': 'aPTT (LAB:LONIC:14979-9)',
            'LAB::LG1777-4': 'Protein (LAB:LONIC:LG1777-4)',
            'LAB::LG32863-9': 'Lymphocytes (LAB:LONIC:LG32863-9)',
            'MED:ATC:C03CA': 'Sulfonamides, plain (MED:ATC:C03CA)',
            'LAB::770-8': 'Neutrophils (LAB:LONIC:770-8)',
            'LAB::5902-2': 'Prothrombin time (PT) (LAB:LONIC:5902-2)',
            'MED:ATC:B01AB': 'Heparin group (MED:ATC:B01AB)',
            'LAB::20570-8': 'Hematocrit (LAB:LONIC:20570-8)',
            'LAB::LG32849-8': 'Eosinophils (LAB:LONIC:LG32849-8)',
            'LAB::LG6039-4': 'Lactate (LAB:LONIC:LG6039-4)',
            'LAB::32623-1': 'Platelet mean volume (LAB:LONIC:32623-1)',
            'LAB::LG6033-7': 'Aspartate aminotransferase (LAB:LONIC:LG6033-7)',
            'LAB::21000-5': 'Erythrocyte distribution width (LAB:LONIC:21000-5)',
            'PX:CH:96361': 'Intravenous infusion, hydration (PX:CH:96361)',
            'LAB::LG32846-4': 'Granulocytes (LAB:LONIC:LG32846-4)',
            'LAB::LG32886-0': 'Neutrophils (LAB:LONIC:LG32886-0)',
            'PX:CH:J1940': 'Injection, furosemide (PX:CH:J1940)',
            'LAB::787-2': 'MCV (LAB:LONIC:787-2)',
            'LAB::61151-7': 'Albumin (LAB:LONIC:61151-7)',
            'DX:09:276bt6': 'Disorders of fluid electrolyte and acid-base balance (DX:09:276bt6)',
            'LAB::LG49829-1': 'Albumin (LAB:LONIC:LG49829-1)',
            'PX:CH:94002': 'Ventilation assist and management (PX:CH:94002)',
            'LAB::890-4': 'Blood group antibody screen (LAB:LONIC:890-4)',
            'PX:CH:71010': 'Radiologic examination, chest (PX:CH:71010)',
            'LAB::776-5': 'Platelet mean volume (LAB:LONIC:776-5)',
            'LAB::LG50024-5': 'Creatinine (LAB:LOINC:LG50024-5)',
            'LAB::LG49881-2': 'Glucose (LAB:LOINC:LG49881-2)',
            'MED:ATC:J01XA': 'Glycopeptide antibacterials (MED:ATC:J01XA)',
            'MED:ATC:H02AB': 'Glucocorticoid (MED:ATC:H02AB)',
            'MED:ATC:B01AC': 'Platelet aggregation inhibitors (MED:ATC:B01AC)',
            'MED:ATC:J01CA': 'Penicillin (MED:ATC:J01CA)',
            'LAB::26505-8': 'Segmented neutrophils (LAB:LOINC:26505-8)',
            'LAB::LG47183-5': 'Vancomycin (LAB:LOINC:LG47183-5)',
            'LAB::6301-6': 'INR in Platelet poor plasma (LAB:LOINC:6301-6)',
            'PX:CH:J1100': 'Dexamethasone Sodium Phosphate (PX:CH:J1100)',
            'PX:CH:J3490': 'Unclassified drugs (PX:CH:J3490)',
            'PX:CH:76770': 'Diagnostic Ultrasound Procedures (PX:CH:J3490)',
            'LAB::LG5903-2': 'Magnesium (LAB:LOINC:LG5903-2)',
            'LAB::LG6426-3': 'Phosphate (LAB:LOINC:LG6426-3)',
            'MED:ATC:C09AA': 'ACE inhibitors, plain (MED:ATC:C09AA)'
        }
        return extracted_data.get(label, label)
    
    def custom_translate_omop_2022_2_no_lonic(self, label):
        return self.custom_translate_omop_2022_2(label).rsplit(" ", 1)[0]
 
    def custom_translate_omop_2022_2_fig2(self, label):
        label0 = label.split('(')[0]
        label_trans = self.custom_translate_omop_2022_2(label0)        
        if len(label.split('('))>1:
            units = '['+label.split('(')[1].replace(')',']')
#            label_trans = label_trans + ' '+units
            label_trans = label_trans.split('(')[0].rstrip()  + units + '\n' + '(' + label_trans.split('(')[1]
        else:
            if len(label_trans.split('('))>1:            
                label_trans = label_trans.split('(')[0].rstrip() + '\n' + '(' + label_trans.split('(')[1]
        return label_trans
    
    def custom_translate_omop_2022_2_outtable(self, label):
        label0 = label.split('(')[0]
        label_trans = self.custom_translate_omop_2022_2(label0)        
        if len(label.split('('))>1:
            units = '['+label.split('(')[1].replace(')',']')
#            label_trans = label_trans + ' '+units
            label_trans = label_trans.split('(')[0].rstrip()  + units  + ' (' + label_trans.split('(')[1]
        else:
            if len(label_trans.split('('))>1:            
                label_trans = label_trans.split('(')[0].rstrip() +  ' (' + label_trans.split('(')[1]
        return label_trans
    
