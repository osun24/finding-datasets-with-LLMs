# Version 1
System Message
You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure.

User Message
A clinical non-small cell lung cancer (NSCLC) dataset, its title, and description are included below.

Inclusion Criteria:
1. Presence of patient-specific survival time and status (e.g., overall survival or progression-free survival).
2. At least one of the following: sample composition of at least 50% stage I patients; or explicit adjuvant chemotherapy annotation.

Let's think step by step and answer the following questions, matching the JSON schema keys:
        1.      q1_survival_data: Does the dataset include survival status and time data (e.g., OS, RFS, PFS)? If so, specify.
        2.      q2_stage_i_or_act: Does it include a majority of patients (>50%) with Stage I NSCLC or explicit adjuvant chemotherapy (ACT) annotation? If so, specify.
        3.      q3_inclusion_justification: Based on the criteria, justify whether the dataset should be considered for inclusion.
        4.      q4_include_in_meta_analysis: Should this dataset be considered for inclusion in the meta-analysis? Answer "Yes" or "No".

## Sources of misclassification:
- No patient-specific survival status and time variables  present.
- Insufficient stage I (23/63). Human reviewers counted total not just among tumor samples. The study has COPD controls.
- LLM counting wrong. (Perhaps can be mitigated with tool use.)

# Version 2
System Message
You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure.

User Message
A clinical non-small cell lung cancer (NSCLC) dataset, its title, and description are included below.

Inclusion Criteria:
1. Presence of patient-specific variables for survival time and status (e.g., overall survival status and overall survival survival time; or progression-free survival status and progression-free survival time).
2. At least one of the following: total-study sample composition of at least 50% stage I patients; or explicit adjuvant chemotherapy annotation.

Let's think step by step and answer the following questions, matching the JSON schema keys:
        1.      q1_survival_data: Does the dataset include patient-specific **variables** for survival status and survival time (e.g., OS, RFS, PFS)? If so, specify the variables.
        2.      q2_stage_i_or_act: Does the dataset, as a whole, include a majority of patients (>50%) with Stage I NSCLC or explicit adjuvant chemotherapy (ACT) annotation? If so, specify.
        3.      q3_inclusion_justification: Based on the criteria, justify whether the dataset should be considered for inclusion.
        4.      q4_include_in_meta_analysis: Should this dataset be considered for inclusion in the meta-analysis? Answer "Yes" or "No".

## Notes:
- GPT-4.1-mini for samples_table_GSE244645.csv: "T1N0M0" repeat 4657 times resulting in failed response
