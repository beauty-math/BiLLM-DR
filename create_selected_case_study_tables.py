
import os, csv
from collections import OrderedDict

OUT_DIR = '/home/zqgaopengfei/project/zym/results/case_study_verification/all_selected_diseases'
os.makedirs(OUT_DIR, exist_ok=True)

DISEASES = OrderedDict([
    ('600807', ('Asthma / asthma-related traits', 'asthma')),
    ('157300', ('Migraine', 'migraine')),
    ('125853', ('Type 2 diabetes', 'type2_diabetes')),
    ('181500', ('Schizophrenia susceptibility', 'schizophrenia')),
    ('165720', ('Osteoarthritis susceptibility', 'osteoarthritis')),
])

# Top-10 rows copied from the generated case-study outputs on 2026-05-12.
TOPS = {
('Fdataset','600807'): [
(1,'Prednisone','DB00635',0.994864,'DrugBank, PubChem, ClinicalTrials','Direct asthma evidence: systemic glucocorticoid; DrugBank indication/trials include asthma and PubChem provides DrugBank-derived corticosteroid information.'),
(2,'Dexamethasone','DB01234',0.768629,'DrugBank, PubChem','Corticosteroid used for inflammatory/respiratory conditions; support is pharmacologic and indication-related.'),
(3,'Prednisolone','DB00860',0.685211,'DrugBank, PubChem','Glucocorticoid anti-inflammatory agent; indication-related asthma/airway inflammation evidence.'),
(4,'Methylprednisolone','DB00959',0.541824,'DrugBank, PubChem','DrugBank/label-derived evidence supports respiratory/allergic disorders and asthma-related use.'),
(5,'Ciclesonide','DB01410',0.439655,'DrugBank, PubChem','Inhaled corticosteroid used for obstructive airway disease/asthma.'),
(6,'Ciprofloxacin','DB00537',0.424924,'NA','No direct asthma therapeutic evidence in the target databases.'),
(7,'Zafirlukast','DB00549',0.362018,'DrugBank, PubChem, ClinicalTrials','Leukotriene receptor antagonist used for prophylaxis/chronic treatment of asthma.'),
(8,'Trimethoprim','DB00440',0.232842,'NA','Antibacterial; no direct asthma therapeutic evidence found.'),
(9,'Levofloxacin','DB01137',0.192709,'NA','Antibacterial; no direct asthma therapeutic evidence found.'),
(10,'Promethazine','DB01069',0.125778,'NA','Antihistamine/antiemetic; no direct asthma therapeutic evidence found.'),
],
('Cdataset','600807'): [
(1,'Prednisone','DB00635',0.999983,'DrugBank, PubChem, ClinicalTrials','Direct asthma evidence.'),
(2,'Zafirlukast','DB00549',0.999944,'DrugBank, PubChem, ClinicalTrials','Direct asthma evidence; LTRA for asthma maintenance.'),
(3,'Hydrocortisone','DB00741',0.999841,'DrugBank, PubChem','Corticosteroid anti-inflammatory evidence; asthma-related steroid class support.'),
(4,'Triamcinolone','DB00620',0.999805,'DrugBank, PubChem','Corticosteroid used for inflammatory/allergic airway-related conditions.'),
(5,'Dexamethasone','DB01234',0.999736,'DrugBank, PubChem','Corticosteroid anti-inflammatory/respiratory indication-related evidence.'),
(6,'Ciclesonide','DB01410',0.999649,'DrugBank, PubChem','Inhaled corticosteroid used for obstructive airway disease/asthma.'),
(7,'Moxifloxacin','DB00218',0.999644,'NA','Antibacterial; no direct asthma therapeutic evidence found.'),
(8,'Methylprednisolone','DB00959',0.999547,'DrugBank, PubChem','Steroid evidence for respiratory/allergic disorders and asthma-related use.'),
(9,'Ciprofloxacin','DB00537',0.999327,'NA','Antibacterial; no direct asthma therapeutic evidence found.'),
(10,'Levofloxacin','DB01137',0.991330,'NA','Antibacterial; no direct asthma therapeutic evidence found.'),
],
('Fdataset','157300'): [
(1,'Ibuprofen','DB01050',0.944170,'DrugBank, PubChem','NSAID with headache/migraine evidence in drug information resources.'),
(2,'Metoprolol','DB00264',0.907299,'DrugBank, PubChem','Beta-blocker with migraine-prevention evidence cited in DrugBank references/drug information.'),
(3,'Methylprednisolone','DB00959',0.864825,'NA','Steroid; not treated as direct migraine evidence in the target database table.'),
(4,'Fluoxetine','DB00472',0.720149,'NA','Antidepressant; no direct migraine indication evidence retained.'),
(5,'Valproic acid','DB00313',0.651331,'DrugBank, PubChem','Anticonvulsant with migraine/migraine-prevention evidence.'),
(6,'Buclizine','DB00354',0.589784,'NA','Antihistamine/antiemetic; no direct migraine evidence retained.'),
(7,'Diclofenac','DB00586',0.524906,'DrugBank, PubChem','NSAID with migraine/headache evidence and pain/inflammation indication.'),
(8,'Labetalol','DB00598',0.505601,'NA','Antihypertensive; no direct migraine evidence retained.'),
(9,'Phenytoin','DB00252',0.497707,'NA','Anticonvulsant; no direct migraine evidence retained.'),
(10,'Meprobamate','DB00371',0.405416,'NA','Sedative/anxiolytic; no direct migraine evidence retained.'),
],
('Cdataset','157300'): [
(1,'Misoprostol','DB00929',0.999704,'NA','No direct migraine therapeutic evidence found.'),
(2,'Clonidine','DB00575',0.999611,'NA','No direct migraine evidence retained.'),
(3,'Ibuprofen','DB01050',0.999536,'DrugBank, PubChem','NSAID with headache/migraine evidence.'),
(4,'Salsalate','DB01399',0.998523,'NA','NSAID/salicylate; no direct migraine evidence retained.'),
(5,'Labetalol','DB00598',0.996812,'NA','No direct migraine evidence retained.'),
(6,'Metoprolol','DB00264',0.991383,'DrugBank, PubChem','Migraine-prevention evidence.'),
(7,'Dicyclomine','DB00804',0.983760,'NA','Antispasmodic; no direct migraine evidence.'),
(8,'Hyoscyamine','DB00424',0.978901,'NA','Anticholinergic; no direct migraine evidence.'),
(9,'Buclizine','DB00354',0.955523,'NA','No direct migraine evidence retained.'),
(10,'Tramadol','DB00193',0.950154,'NA','Analgesic but no disease-specific migraine evidence retained for this table.'),
],
('Fdataset','125853'): [
(1,'Losartan','DB00678',0.004224,'DrugBank, PubChem','Indirect diabetes evidence: diabetic nephropathy in type 2 diabetes with hypertension.'),
(2,'Telmisartan','DB00966',0.003630,'DrugBank, PubChem','Indirect diabetes/metabolic or diabetic nephropathy-related evidence.'),
(3,'Cimetidine','DB00501',0.003621,'NA','No direct type 2 diabetes evidence retained.'),
(4,'Orlistat','DB01083',0.002540,'NA','Obesity/metabolic relevance but no direct type 2 diabetes evidence retained.'),
(5,'Folinic acid','DB00650',0.001312,'NA','No direct type 2 diabetes evidence retained.'),
(6,'Liothyronine','DB00279',0.001118,'NA','Thyroid hormone; no direct type 2 diabetes evidence retained.'),
(7,'Gemfibrozil','DB01241',0.001005,'NA','Lipid-lowering; no direct type 2 diabetes evidence retained.'),
(8,'Iloprost','DB01088',0.001005,'NA','No direct type 2 diabetes evidence retained.'),
(9,'Quinapril','DB00881',0.000412,'DrugBank, PubChem','Indirect diabetes evidence through diabetic nephropathy/hypertension-related use.'),
(10,'Alanine','DB00160',0.000390,'NA','No direct type 2 diabetes evidence retained.'),
],
('Cdataset','125853'): [
(1,'Canagliflozin','DB08907',0.999915,'DrugBank, PubChem, ClinicalTrials','Direct type 2 diabetes evidence: SGLT2 inhibitor approved/used for type 2 diabetes; clinical trials in T2DM.'),
(2,'Octreotide','DB00104',0.745317,'NA','No direct type 2 diabetes therapeutic evidence retained.'),
(3,'Levothyroxine','DB00451',0.449537,'NA','Thyroid hormone; no direct type 2 diabetes evidence retained.'),
(4,'Nadolol','DB01203',0.044969,'NA','Antihypertensive; no direct type 2 diabetes evidence retained.'),
(5,'Liothyronine','DB00279',0.012406,'NA','No direct type 2 diabetes evidence retained.'),
(6,'Methimazole','DB00763',0.011542,'NA','Antithyroid drug; no direct type 2 diabetes evidence retained.'),
(7,'Clomifene/Raloxifene-like SERM','DB00882',0.004057,'NA','No direct type 2 diabetes evidence retained.'),
(8,'Drospirenone','DB01395',0.002626,'NA','No direct type 2 diabetes evidence retained.'),
(9,'Hydrochlorothiazide','DB00999',0.002138,'NA','Antihypertensive/diuretic; no direct type 2 diabetes treatment evidence retained.'),
(10,'Zonisamide','DB00909',0.002077,'NA','No direct type 2 diabetes evidence retained.'),
],
('Fdataset','181500'): [
(1,'Citalopram','DB00215',0.664408,'NA','Antidepressant; no direct schizophrenia evidence retained.'),
(2,'Clonidine','DB00575',0.648169,'NA','No direct schizophrenia evidence retained.'),
(3,'Prochlorperazine','DB00433',0.349400,'DrugBank, PubChem, ClinicalTrials','Phenothiazine antipsychotic; DrugBank/PubChem mention schizophrenia/psychotic disorders.'),
(4,'Lisdexamfetamine','DB01255',0.259138,'NA','Stimulant; no direct schizophrenia evidence retained.'),
(5,'Venlafaxine','DB00285',0.243859,'NA','Antidepressant; no direct schizophrenia evidence retained.'),
(6,'Nortriptyline','DB00540',0.240493,'NA','Antidepressant; no direct schizophrenia evidence retained.'),
(7,'Perphenazine','DB00508',0.125649,'DrugBank, PubChem','Phenothiazine antipsychotic; used for psychotic disorders/schizophrenia.'),
(8,'Paroxetine','DB00715',0.085822,'NA','Antidepressant; no direct schizophrenia evidence retained.'),
(9,'Buspirone','DB00490',0.055154,'NA','Anxiolytic; no direct schizophrenia evidence retained.'),
(10,'Topiramate','DB00273',0.049292,'NA','Antiepileptic; no direct schizophrenia evidence retained.'),
],
('Cdataset','181500'): [
(1,'Baclofen','DB00181',0.999999,'NA','Muscle relaxant; no direct schizophrenia evidence retained.'),
(2,'Clonidine','DB00575',0.999992,'NA','No direct schizophrenia evidence retained.'),
(3,'Prochlorperazine','DB00433',0.999754,'DrugBank, PubChem, ClinicalTrials','Direct antipsychotic/schizophrenia evidence.'),
(4,'Tetrabenazine','DB04844',0.999611,'NA','VMAT2 inhibitor for hyperkinetic disorders; no direct schizophrenia evidence retained.'),
(5,'Perphenazine','DB00508',0.998911,'DrugBank, PubChem','Direct antipsychotic/schizophrenia evidence.'),
(6,'Carbidopa','DB00190',0.993005,'NA','Parkinson-related adjunct; no direct schizophrenia evidence.'),
(7,'Fluvoxamine','DB00176',0.991979,'NA','SSRI/OCD evidence; no direct schizophrenia evidence retained.'),
(8,'Biperiden','DB00810',0.983284,'NA','Antiparkinson/EPS therapy; no direct schizophrenia evidence retained.'),
(9,'Clobazam','DB00349',0.983148,'NA','Benzodiazepine anticonvulsant; no direct schizophrenia evidence retained.'),
(10,'Lorazepam','DB00186',0.975750,'NA','Benzodiazepine supportive use only; no direct schizophrenia evidence retained.'),
],
('Fdataset','165720'): [
(1,'Nabumetone','DB00461',0.642339,'DrugBank, PubChem, ClinicalTrials','NSAID indicated for symptomatic relief of osteoarthritis.'),
(2,'Methylprednisolone','DB00959',0.332406,'DrugBank, PubChem','Steroid; DrugBank notes synovitis of osteoarthritis/anti-inflammatory use.'),
(3,'Misoprostol','DB00929',0.225413,'DrugBank, ClinicalTrials','Related OA evidence through NSAID-associated ulcer prevention/DrugBank OA trial listing; not disease-modifying.'),
(4,'Dexamethasone','DB01234',0.151849,'DrugBank, PubChem','Anti-inflammatory corticosteroid; osteoarthritis-related symptomatic/inflammatory evidence.'),
(5,'Acetaminophen','DB00316',0.093303,'DrugBank, PubChem','Analgesic evidence for pain management; relevant to OA symptom relief.'),
(6,'Salsalate','DB01399',0.065549,'DrugBank, PubChem','NSAID/salicylate used for osteoarthritis and rheumatic disorders.'),
(7,'Caffeine','DB00201',0.041927,'NA','No direct osteoarthritis evidence retained.'),
(8,'Alendronic acid','DB00630',0.027113,'NA','Osteoporosis/bone disease evidence; no direct osteoarthritis evidence retained.'),
(9,'Prednisone','DB00635',0.010077,'DrugBank, PubChem','Steroid anti-inflammatory evidence; retained as related symptomatic evidence.'),
(10,'Meclofenamic acid','DB00939',0.009156,'DrugBank, PubChem','NSAID/anti-inflammatory evidence relevant to arthritic pain/inflammation.'),
],
('Cdataset','165720'): [
(1,'Misoprostol','DB00929',0.977482,'DrugBank, ClinicalTrials','OA-related evidence through NSAID ulcer-prevention/trial listing; indirect symptomatic-context evidence.'),
(2,'Bromfenac','DB00963',0.892823,'NA','Ophthalmic NSAID; no direct osteoarthritis evidence retained.'),
(3,'Caffeine','DB00201',0.743027,'NA','No direct osteoarthritis evidence retained.'),
(4,'Acetaminophen','DB00316',0.276833,'DrugBank, PubChem','Analgesic evidence relevant to OA symptom relief.'),
(5,'Butalbital','DB00241',0.019343,'NA','No direct osteoarthritis evidence retained.'),
(6,'Codeine','DB00318',0.018849,'DrugBank, PubChem','Analgesic evidence for pain; related to symptom relief rather than OA-specific therapy.'),
(7,'Meperidine','DB00454',0.007570,'NA','Opioid analgesic; no OA-specific evidence retained.'),
(8,'Ketorolac','DB00465',0.005888,'DrugBank, PubChem','NSAID evidence includes osteoarthritis/pain-related indications.'),
(9,'Ethanol','DB00898',0.004110,'NA','No direct osteoarthritis evidence retained.'),
(10,'Methadone','DB00333',0.002841,'NA','Opioid analgesic; no OA-specific evidence retained.'),
],
}

source_urls = {
'DrugBank': 'https://go.drugbank.com/',
'PubChem': 'https://pubchem.ncbi.nlm.nih.gov/',
'DrugCentral': 'https://drugcentral.org/',
'ClinicalTrials': 'https://clinicaltrials.gov/',
}

def safe_name(s):
    return s.replace(' ', '_').replace('/', '_').replace('-', '_')

combined_rows=[]
combined_md=[]
for (dataset, omim), rows in TOPS.items():
    disease, slug = DISEASES[omim]
    stem=f'{dataset}_{omim}_{slug}_top10'
    paper_path=os.path.join(OUT_DIR, stem+'_paper_style.csv')
    detail_path=os.path.join(OUT_DIR, stem+'_evidence_detail.csv')
    md_path=os.path.join(OUT_DIR, stem+'_table.md')
    with open(paper_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['Rank','Candidate drugs (DrugBank IDs)','Pieces of evidence'])
        for rank,drug,dbid,score,evidence,note in rows:
            w.writerow([rank, f'{drug} ({dbid})', evidence])
            combined_rows.append([dataset,omim,disease,rank,drug,dbid,score,evidence,note])
    with open(detail_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['Dataset','OMIM','Disease','Rank','Drug','DrugBank_ID','Prediction_score','Pieces_of_evidence','Evidence_note','DrugBank_URL','PubChem_Search','DrugCentral_Search','ClinicalTrials_Search'])
        for rank,drug,dbid,score,evidence,note in rows:
            w.writerow([dataset,omim,disease,rank,drug,dbid,score,evidence,note,
                        f'https://go.drugbank.com/drugs/{dbid}',
                        f'https://pubchem.ncbi.nlm.nih.gov/#query={drug.replace(" ", "%20")}',
                        f'https://drugcentral.org/?q={drug.replace(" ", "+")}',
                        f'https://clinicaltrials.gov/search?term={drug.replace(" ", "%20")}%20{disease.split(" /")[0].replace(" ", "%20")}'])
    md = []
    md.append(f"TABLE: The top 10 DReKGNN-predicted candidate drugs for {disease} ({dataset}). 'NA' denotes no evidence.\n")
    md.append('| Rank | Candidate drugs (DrugBank IDs) | Pieces of evidence |')
    md.append('|---:|---|---|')
    for rank,drug,dbid,score,evidence,note in rows:
        md.append(f'| {rank} | {drug} ({dbid}) | {evidence} |')
    md.append('\nNote: CTD was not included because automated access was blocked; add CTD manually only when directly confirmed. Evidence is intentionally conservative.\n')
    text='\n'.join(md)
    with open(md_path,'w',encoding='utf-8') as f:
        f.write(text)
    combined_md.append(text)

with open(os.path.join(OUT_DIR,'all_selected_case_study_evidence_detail.csv'),'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['Dataset','OMIM','Disease','Rank','Drug','DrugBank_ID','Prediction_score','Pieces_of_evidence','Evidence_note'])
    w.writerows(combined_rows)
with open(os.path.join(OUT_DIR,'all_selected_case_study_tables.md'),'w',encoding='utf-8') as f:
    f.write('\n\n'.join(combined_md))

print('Saved files to:', OUT_DIR)
for name in sorted(os.listdir(OUT_DIR)):
    print(name)
