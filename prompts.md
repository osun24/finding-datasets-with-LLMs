# GEO
Prompt version: v2

System message:

You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure.

User prompt template:

A clinical non-small cell lung cancer (NSCLC) dataset, its title, and description are included below.

Inclusion Criteria:
1. Presence of patient-specific variables for survival time and status (e.g., overall survival status and overall survival survival time; or progression-free survival status and progression-free survival time).
2. At least one of the following: total-study sample composition of at least 50% stage I patients; or explicit adjuvant chemotherapy annotation.

Let's think step by step and answer the following questions, matching the JSON schema keys:
        1.      q1_survival_data: Does the dataset include patient-specific **variables** for survival status and survival time (e.g., OS, RFS, PFS)? If so, specify the variables.
        2.      q2_stage_i_or_act: Does the dataset, as a whole, include a majority of patients (>50%) with Stage I NSCLC or explicit adjuvant chemotherapy (ACT) annotation? If so, specify.
        3.      q3_inclusion_justification: Based on the criteria, justify whether the dataset should be considered for inclusion.
        4.      q4_include_in_meta_analysis: Should this dataset be considered for inclusion in the meta-analysis? Answer "Yes" or "No".

{CSV_DATASET_CONTENT}

JSON response_format:
```JSON
{
  "type": "json_schema",
  "json_schema": {
    "name": "geo_dataset_screening",
    "description": "Structured screening decision for GEO NSCLC meta-analysis datasets.",
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "q1_survival_data",
        "q2_stage_i_or_act",
        "q3_inclusion_justification",
        "q4_include_in_meta_analysis"
      ],
      "properties": {
        "q1_survival_data": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q2_stage_i_or_act": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "stage_i_evidence",
            "act_evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "stage_i_evidence": {
              "type": "string"
            },
            "act_evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q3_inclusion_justification": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "meets_survival_criterion",
            "meets_stage_or_act_criterion",
            "justification"
          ],
          "properties": {
            "meets_survival_criterion": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "meets_stage_or_act_criterion": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q4_include_in_meta_analysis": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No"
              ]
            }
          }
        }
      }
    },
    "strict": true
  }
}
```

# ProteomeXChange
================================================================================
Prompt version: v2

System message:

You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure.

User prompt template:

A dataset's title and description are included below.

Inclusion Criteria: 
1. The dataset must be about ovarian cancer.
2. The dataset must contain healthy patients as controls OR involve diagnostic biomarker discovery for ovarian cancer.
        2a. If explicit sample compositions are provided, the dataset must contain at least 20% healthy patients as controls. If explicit sample compositions are not provided but healthy patient controls are present, include the dataset for additional human review.
3. The dataset must be on proteomics.

Exclusion Criteria:
4. Non-clinical datasets (e.g., xenografts, cell lines, animal models, and ex vivo studies) are excluded.

Let's think step by step and answer the following questions, matching the JSON schema keys:
        1.      q1_ovarian_cancer: Is the dataset about ovarian cancer? If so, justify.
        2.      q2_healthy_controls_or_biomarker_discovery: Does the dataset contain healthy patients as controls or involve diagnostic biomarker discovery for ovarian cancer? If so, please explain.
        2a.     q2a_healthy_control_composition: If the explicit sample compositions are provided, does the dataset contain at least 20% healthy patients as controls?
 If so, please explain.
        3.      q3_proteomics: Does it include proteomics? If so, please explain.
        4.      q4_non_clinical: Does the dataset include non-clinical samples (e.g., xenografts, cell lines, animal models, ex vivo studies, and in vitro studies)? If so, please explain.
        5.      q5_inclusion_justification: Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.
        6.      q6_include_dataset: Should the dataset be considered for inclusion? (Answer "Yes" or "No")

{TITLE_AND_DESCRIPTION}

JSON response_format:
```JSON
{
  "type": "json_schema",
  "json_schema": {
    "name": "proteomexchange_dataset_screening",
    "description": "Structured screening decision for ProteomeXchange ovarian cancer proteomics datasets.",
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "q1_ovarian_cancer",
        "q2_healthy_controls_or_biomarker_discovery",
        "q2a_healthy_control_composition",
        "q3_proteomics",
        "q4_non_clinical",
        "q5_inclusion_justification",
        "q6_include_dataset"
      ],
      "properties": {
        "q1_ovarian_cancer": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q2_healthy_controls_or_biomarker_discovery": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q2a_healthy_control_composition": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear",
                "Not applicable"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q3_proteomics": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q4_non_clinical": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q5_inclusion_justification": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "meets_ovarian_cancer_criterion",
            "meets_healthy_controls_criterion",
            "meets_proteomics_criterion",
            "justification"
          ],
          "properties": {
            "meets_ovarian_cancer_criterion": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "meets_healthy_controls_criterion": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "meets_proteomics_criterion": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q6_include_dataset": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No"
              ]
            }
          }
        }
      }
    },
    "strict": true
  }
}
```


# BioStudies/ArrayExpress
--------------------------------------------------------------------------------
Prompt for target disease: AD
--------------------------------------------------------------------------------

System message:

You are a neurodegeneration transcriptomics expert evaluating clinical datasets, based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure.

User prompt template:

An ArrayExpress/BioStudies dataset's context is included below.

Inclusion Criteria:
1. The dataset must be about human microarray expression profiling.
2. The dataset must pertain to Alzheimer's disease (AD).
3. The dataset must involve samples from CNS regions relevant to Alzheimer's disease (AD).

Exclusion Criteria:
4. If the dataset involves patient-derived in vitro cell lines or disease models, exclude it.
5. If the study design is not case/control, exclude it.
6. If the brain regions sampled are not significantly affected by AD neurodegeneration (e.g., cerebellum), exclude it.
7. If the dataset only involves intermediate AD phenotypes, exclude it. Only disease AD samples classified as Braak V/VI (corresponding to a neocortical NFT stage) should be included.
8. Familial AD datasets are excluded. Sporadic AD datasets are included.
9. Technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics, are excluded.

Let's think step by step and answer the following questions, matching the JSON schema keys:
        1.      q1_human_microarray_expression_profiling: Is the dataset about human microarray expression profiling? If so, please justify.
        2.      q2_target_disease: Does the dataset pertain to Alzheimer's disease (AD)? If so, please justify.
        3.      q3_human_cns_tissue: Does the dataset involve samples from human brain/CNS tissue? If so, please justify.
        4.      q4_non_clinical: Does the dataset involve patient-derived in vitro cell lines or disease models? If so, please justify.
        5.      q5_case_control_design: Does the dataset have a case/control study design? If so, please justify.
        6.      q6_irrelevant_brain_region: Does the dataset involve brain regions not significantly affected by AD neurodegeneration (e.g., cerebellum)? If so, please justify.
        7.      q7_intermediate_phenotype: Does the dataset only involve intermediate AD phenotypes (e.g., Braak III/IV)? If so, please justify.
        8.      q8_familial_alzheimer_disease: Does the dataset only involve familial AD cases? If so, please justify.
        9.      q9_incompatible_technology: Does the dataset involve technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics? If so, please justify.

        10.     q10_inclusion_justification: Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.
        11.     q11_include_dataset: Should the dataset be considered for inclusion? (Answer "Yes" or "No")

{ARRAYEXPRESS_CONTEXT_TEXT}

--------------------------------------------------------------------------------
Prompt for target disease: LBD
--------------------------------------------------------------------------------

System message:

You are a neurodegeneration transcriptomics expert evaluating clinical datasets, based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure.

User prompt template:

An ArrayExpress/BioStudies dataset's context is included below.

Inclusion Criteria:
1. The dataset must be about human microarray expression profiling.
2. The dataset must pertain to Lewy body diseases (LBD).
3. The dataset must involve samples from CNS regions relevant to Lewy body diseases (LBD).

Exclusion Criteria:
4. If the dataset involves patient-derived in vitro cell lines or disease models, exclude it.
5. If the study design is not case/control, exclude it.
6. If the brain regions sampled are not significantly affected by LBD neurodegeneration, exclude it.
7. Familial LBD datasets are excluded. Sporadic LBD datasets are included.
8. Technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics, are excluded.

Let's think step by step and answer the following questions, matching the JSON schema keys:
        1.      q1_human_microarray_expression_profiling: Is the dataset about human microarray expression profiling? If so, please justify.
        2.      q2_target_disease: Does the dataset pertain to Lewy body diseases (LBD)? If so, please justify.
        3.      q3_human_cns_tissue: Does the dataset involve samples from human brain/CNS tissue? If so, please justify.
        4.      q4_non_clinical: Does the dataset involve patient-derived in vitro cell lines or disease models? If so, please justify.
        5.      q5_case_control_design: Does the dataset have a case/control study design? If so, please justify.
        6.      q6_irrelevant_brain_region: Does the dataset involve brain regions not significantly affected by LBD neurodegeneration? If so, please justify.
        7.      q7_familial_disease: Does the dataset only involve familial LBD cases? If so, please justify.
        8.      q8_incompatible_technology: Does the dataset involve technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics? If so, please justify.

        9.      q9_inclusion_justification: Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.
        10.     q10_include_dataset: Should the dataset be considered for inclusion? (Answer "Yes" or "No")

{ARRAYEXPRESS_CONTEXT_TEXT}

--------------------------------------------------------------------------------
Prompt for target disease: ALS-FTD
--------------------------------------------------------------------------------

System message:

You are a neurodegeneration transcriptomics expert evaluating clinical datasets, based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure.

User prompt template:

An ArrayExpress/BioStudies dataset's context is included below.

Inclusion Criteria:
1. The dataset must be about human microarray expression profiling.
2. The dataset must pertain to amyotrophic lateral sclerosis-frontotemporal dementia (ALS-FTD).
3. The dataset must involve samples from CNS regions relevant to ALS-FTD.

Exclusion Criteria:
4. If the dataset involves patient-derived in vitro cell lines or disease models, exclude it.
5. If the study design is not case/control, exclude it.
6. If the brain regions sampled are not significantly affected by ALS-FTD neurodegeneration, exclude it.
7. Technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics, are excluded.

Let's think step by step and answer the following questions, matching the JSON schema keys:
        1.      q1_human_microarray_expression_profiling: Is the dataset about human microarray expression profiling? If so, please justify.
        2.      q2_target_disease: Does the dataset pertain to amyotrophic lateral sclerosis-frontotemporal dementia (ALS-FTD)? If so, please justify.
        3.      q3_human_cns_tissue: Does the dataset involve samples from human brain/CNS tissue? If so, please justify.
        4.      q4_non_clinical: Does the dataset involve patient-derived in vitro cell lines or disease models? If so, please justify.
        5.      q5_case_control_design: Does the dataset have a case/control study design? If so, please justify.
        6.      q6_irrelevant_brain_region: Does the dataset involve brain regions not significantly affected by ALS-FTD neurodegeneration? If so, please justify.
        7.      q7_incompatible_technology: Does the dataset involve technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics? If so, please justify.

        8.      q8_inclusion_justification: Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.
        9.      q9_include_dataset: Should the dataset be considered for inclusion? (Answer "Yes" or "No")

{ARRAYEXPRESS_CONTEXT_TEXT}

================================================================================

JSON response_format for AD:
```JSON
{
  "type": "json_schema",
  "json_schema": {
    "name": "arrayexpress_ad_dataset_screening",
    "description": "Structured screening decision targeting AD for ArrayExpress microarray datasets.",
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "q1_human_microarray_expression_profiling",
        "q2_target_disease",
        "q3_human_cns_tissue",
        "q4_non_clinical",
        "q5_case_control_design",
        "q6_irrelevant_brain_region",
        "q7_intermediate_phenotype",
        "q8_familial_alzheimer_disease",
        "q9_incompatible_technology",
        "q10_inclusion_justification",
        "q11_include_dataset"
      ],
      "properties": {
        "q1_human_microarray_expression_profiling": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q2_target_disease": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q3_human_cns_tissue": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q4_non_clinical": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q5_case_control_design": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q6_irrelevant_brain_region": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q7_intermediate_phenotype": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q8_familial_alzheimer_disease": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q9_incompatible_technology": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q10_inclusion_justification": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "justification"
          ],
          "properties": {
            "justification": {
              "type": "string"
            }
          }
        },
        "q11_include_dataset": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No"
              ]
            }
          }
        }
      }
    },
    "strict": true
  }
}
```
JSON response_format for LBD:
```JSON
{
  "type": "json_schema",
  "json_schema": {
    "name": "arrayexpress_lbd_dataset_screening",
    "description": "Structured screening decision targeting LBD for ArrayExpress microarray datasets.",
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "q1_human_microarray_expression_profiling",
        "q2_target_disease",
        "q3_human_cns_tissue",
        "q4_non_clinical",
        "q5_case_control_design",
        "q6_irrelevant_brain_region",
        "q7_familial_disease",
        "q8_incompatible_technology",
        "q9_inclusion_justification",
        "q10_include_dataset"
      ],
      "properties": {
        "q1_human_microarray_expression_profiling": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q2_target_disease": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q3_human_cns_tissue": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q4_non_clinical": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q5_case_control_design": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q6_irrelevant_brain_region": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q7_familial_disease": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q8_incompatible_technology": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q9_inclusion_justification": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "justification"
          ],
          "properties": {
            "justification": {
              "type": "string"
            }
          }
        },
        "q10_include_dataset": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No"
              ]
            }
          }
        }
      }
    },
    "strict": true
  }
}
```
JSON response_format for ALS-FTD:
```JSON
{
  "type": "json_schema",
  "json_schema": {
    "name": "arrayexpress_als_ftd_dataset_screening",
    "description": "Structured screening decision targeting ALS-FTD for ArrayExpress microarray datasets.",
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "q1_human_microarray_expression_profiling",
        "q2_target_disease",
        "q3_human_cns_tissue",
        "q4_non_clinical",
        "q5_case_control_design",
        "q6_irrelevant_brain_region",
        "q7_incompatible_technology",
        "q8_inclusion_justification",
        "q9_include_dataset"
      ],
      "properties": {
        "q1_human_microarray_expression_profiling": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q2_target_disease": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q3_human_cns_tissue": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q4_non_clinical": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q5_case_control_design": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q6_irrelevant_brain_region": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q7_incompatible_technology": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer",
            "evidence",
            "justification"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No",
                "Unclear"
              ]
            },
            "evidence": {
              "type": "string"
            },
            "justification": {
              "type": "string"
            }
          }
        },
        "q8_inclusion_justification": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "justification"
          ],
          "properties": {
            "justification": {
              "type": "string"
            }
          }
        },
        "q9_include_dataset": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "answer"
          ],
          "properties": {
            "answer": {
              "type": "string",
              "enum": [
                "Yes",
                "No"
              ]
            }
          }
        }
      }
    },
    "strict": true
  }
}
```