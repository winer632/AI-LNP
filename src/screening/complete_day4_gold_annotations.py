"""Complete and validate the manually reviewed Day 4 gold annotations.

The records in this module are frozen human judgments, not model-generated
labels. Running the module is idempotent: existing records with the same IDs
are replaced, while unrelated records are preserved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "annotations" / "gold_v1"
REVIEWER = "human_review_frozen"
REVIEWED_AT = "2026-07-22"


def read_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (GOLD / name).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(name: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    with (GOLD / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def upsert(name: str, key: str, new_rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    old_fields, rows = read_csv(name)
    output_fields = fields or old_fields
    replacements = {row[key]: row for row in new_rows}
    kept = [row for row in rows if row.get(key) not in replacements]
    kept.extend(new_rows)
    write_csv(name, output_fields, kept)


def evidence(
    evidence_id: str,
    paper: str,
    entity_type: str,
    entity_id: str,
    field: str,
    text: str,
    location_type: str,
    section: str,
    source: str,
    *,
    xml_id: str = "",
    page: str = "",
    table: str = "",
    row: str = "",
    column: str = "",
    figure: str = "",
    supplement: str = "",
    method: str = "manual_full_text_review",
    status: str = "verified",
    notes: str = "",
) -> dict[str, str]:
    return {
        "evidence_id": evidence_id,
        "gold_paper_id": paper,
        "supported_entity_type": entity_type,
        "supported_entity_id": entity_id,
        "field_name": field,
        "evidence_text": text,
        "evidence_location_type": location_type,
        "section_name": section,
        "page_number": page,
        "table_number": table,
        "table_row": row,
        "table_column": column,
        "figure_number": figure,
        "supplement_identifier": supplement,
        "xml_file": source,
        "xml_element_id": xml_id,
        "extraction_method": method,
        "review_status": status,
        "reviewer_notes": notes,
    }


def main() -> None:
    # Freeze paper-level resolutions discovered during full-text review.
    fields, papers = read_csv("papers.csv")
    resolutions = {
        "GP-002": ("completed", "Complete SM-102 formulation with direct hepatocyte reporter expression."),
        "GP-003": ("completed_no_eligible_records", "Review article; expected extraction result is zero original experiments."),
        "GP-005": ("completed", "Structured-table case showing high Kupffer uptake but negligible mRNA translation."),
        "GP-007": ("completed_with_incomplete_formulation", "Direct LSEC therapeutic evidence, but lipid identities, ratios, and dose concentration are incomplete."),
        "GP-008": ("completed_with_role_ambiguity", "The LNP transfects CD163-positive macrophages; activated HSCs are therapeutic targets, not delivery recipients."),
        "GP-009": ("completed_no_eligible_records", "Irrelevant HSC acronym hit: hematopoietic stem cells, not hepatic stellate cells."),
    }
    for paper in papers:
        if paper["gold_paper_id"] in resolutions:
            status, note = resolutions[paper["gold_paper_id"]]
            paper["annotation_status"] = status
            paper["reviewer"] = REVIEWER
            paper["reviewed_at"] = REVIEWED_AT
            paper["notes"] = note
    write_csv("papers.csv", fields, papers)

    xml = "data/raw/fulltext/gold_v1/xml/"
    oa = "data/raw/fulltext/oa_packages/"
    new_evidence = [
        evidence("EVID-012", "GP-002", "formulation", "GF-003", "composition", "SM-102:DSPC:cholesterol:DMG-PEG2000 = 50:10:38.5:1.5 molar ratio.", "xml_paragraph", "Introduction", xml + "candidate_00003_PMC13334401.xml", xml_id="p0040"),
        evidence("EVID-013", "GP-002", "experiment", "GX-004", "hepatocyte_reporter_delivery", "Five healthy mice received 10 micrograms eGFP mRNA-LNP by tail vein and were assessed after 24 hours.", "xml_paragraph_and_figure", "Results", xml + "candidate_00003_PMC13334401.xml", xml_id="p0050", figure="Figure 1"),
        evidence("EVID-014", "GP-002", "outcome", "GO-008", "hepatocyte_expression", "Immunohistochemistry showed strong eGFP staining in virtually all hepatocytes.", "xml_paragraph_and_figure", "Results", xml + "candidate_00003_PMC13334401.xml", xml_id="p0050", figure="Figure 1F"),
        evidence("EVID-015", "GP-003", "paper", "GP-003", "original_experiment_status", "The publication describes itself as a state-of-the-art review and does not report an original tested LNP experiment.", "xml_article", "Abstract", xml + "candidate_00019_PMC13184955.xml"),
        evidence("EVID-016", "GP-005", "formulation", "GF-004", "composition", "MC3, DSPC, cholesterol, and DMG-PEG2000 were mixed at 50:10:38.5:1.5 molar ratio in ethanol; RNA was in 0.1 M acetate buffer pH 4.", "xml_paragraph", "Experimental Section", xml + "candidate_00082_PMC11884593.xml"),
        evidence("EVID-017", "GP-005", "formulation", "GF-004", "structured_characterization", "LNP1 containing Egfp mRNA and MC3 measured 130.0 +/- 31.9 nm by NanoSight and 103.5 +/- 1.23 nm by DLS, with zeta potential -4.95 +/- 0.43 mV.", "xml_structured_table", "Results", xml + "candidate_00082_PMC11884593.xml", table="Table 1", row="LNP 1", method="manual_structured_table_review"),
        evidence("EVID-018", "GP-005", "experiment", "GX-005", "kupffer_uptake_and_translation", "Mice received 2 mg/kg Egfp mRNA-LNP intravenously; intravital imaging at 16 hours and F4/80 staining assessed Kupffer-cell translation.", "xml_paragraph_and_figure", "Results", xml + "candidate_00082_PMC11884593.xml", figure="Figure 1"),
        evidence("EVID-019", "GP-005", "outcome", "GO-010", "kupffer_uptake", "Kupffer cells showed high and rapid LNP uptake, but no obvious EGFP-positive F4/80-positive Kupffer cells were detected in vivo.", "xml_paragraph_and_figure", "Results", xml + "candidate_00082_PMC11884593.xml", figure="Figures 1 and 4"),
        evidence("EVID-020", "GP-005", "outcome", "GO-011", "kupffer_translation", "EGFP translation was observed solely in hepatocytes; labeled RNA remained visible as small spots in Kupffer cells after 16 hours.", "xml_paragraph_and_figure", "Results", xml + "candidate_00082_PMC11884593.xml", figure="Figure 4F"),
        evidence("EVID-021", "GP-007", "formulation", "GF-005", "partial_composition", "Cholesterol and DSPE-PEG were dissolved in ethanol with 0.1 mol% FITC-labeled hyaluronic acid; siMicu1 was rapidly mixed using a T-junction, then dialyzed, filtered, and concentrated.", "xml_paragraph", "Methods", xml + "candidate_00167_PMC13137855.xml", status="verified_incomplete", notes="No complete lipid ratio or exact DSPE-PEG identity is reported."),
        evidence("EVID-022", "GP-007", "formulation", "GF-005", "particle_characterization", "siMicu1-LNPs measured 80.03 +/- 1.19 nm, zeta potential 10.12 mV, and 71.97% encapsulation efficiency.", "xml_paragraph_and_figure", "Results", xml + "candidate_00167_PMC13137855.xml", figure="Figure 7A-C"),
        evidence("EVID-023", "GP-007", "experiment", "GX-007", "lsec_treatment", "Mice received siMicu1-LNP intravenously at 5 microliters per gram two hours before HIRI surgery; the siRNA mass concentration was not reported.", "xml_paragraph_and_figure", "Methods and Results", xml + "candidate_00167_PMC13137855.xml", figure="Figure 7E"),
        evidence("EVID-024", "GP-007", "outcome", "GO-013", "lsec_protection", "siMicu1-LNP treatment ameliorated mitochondrial damage, calcium accumulation, and the defenestration phenotype of LSECs after HIRI.", "xml_paragraph_and_figure", "Results", xml + "candidate_00167_PMC13137855.xml", figure="Figure 7H-L"),
        evidence("EVID-025", "GP-008", "formulation", "GF-006", "composition_and_manufacturing", "Ionizable lipid:DSPC:cholesterol:PEG-lipid = 45:30:23.5:1.5. Lipids were mixed with mRNA in 6.25 mM sodium acetate pH 5 at a 3:1 aqueous:organic ratio.", "supplement_pdf_text", "RNA Synthesis and Lipid Nanoparticle Formulation", oa + "PMC13229182/pnas.2534673123.sapp.pdf", page="4", supplement="pnas.2534673123.sapp.pdf", method="manual_visual_pdf_review"),
        evidence("EVID-026", "GP-008", "formulation", "GF-006", "targeting_conjugation", "Anti-CD163 was attached using SATA-maleimide chemistry after DSPE-PEG-maleimide post-insertion; antibody:LNP ratio was 1:20.", "supplement_pdf_text", "RNA Synthesis and Lipid Nanoparticle Formulation", oa + "PMC13229182/pnas.2534673123.sapp.pdf", page="4", supplement="pnas.2534673123.sapp.pdf", method="manual_visual_pdf_review"),
        evidence("EVID-027", "GP-008", "formulation", "GF-006", "pdf_table_characterization", "Appendix Table 1 reports alpha-CD163/LNP diameter 132.05 +/- 3.10 nm and PDI 0.176 +/- 0.017.", "supplement_pdf_table", "Tables S1 to S3", oa + "PMC13229182/pnas.2534673123.sapp.pdf", page="27", table="Appendix Table 1", row="alpha-CD163/LNP", method="manual_visual_pdf_review"),
        evidence("EVID-028", "GP-008", "experiment", "GX-008", "macrophage_delivery", "More than 80% of BMDMs expressed GFP or FAPCAR after alpha-CD163/LNP exposure, compared with fewer than 20% after unmodified LNP.", "xml_paragraph_and_figure", "Results", xml + "candidate_00132_PMC13229182.xml", figure="Figure 1J-M"),
        evidence("EVID-029", "GP-008", "experiment", "GX-009", "hsc_therapeutic_effect", "FAPCAR-expressing macrophages recognized and eliminated FAP-positive activated HSC models; the LNP cargo was expressed in macrophages, not HSCs.", "xml_paragraph_and_figure", "Results and Discussion", xml + "candidate_00132_PMC13229182.xml", figure="Figures 2 and 6"),
        evidence("EVID-030", "GP-008", "outcome", "GO-018", "recipient_cell_specificity", "Supplementary Figure 5 maps reporter expression to CD163/F4/80-positive macrophages and compares ALB, Desmin, F4/80, and SOX9 cell markers.", "supplement_pdf_figure", "Appendix Figure 5", oa + "PMC13229182/pnas.2534673123.sapp.pdf", page="18", figure="Appendix Figure 5G-L", supplement="pnas.2534673123.sapp.pdf", method="manual_visual_pdf_review"),
        evidence("EVID-031", "GP-009", "paper", "GP-009", "target_cell_identity", "HSC denotes hematopoietic stem cells used for bone-marrow-transplant preconditioning, not hepatic stellate cells.", "xml_article", "Abstract", xml + "candidate_00186_PMC12265960.xml"),
    ]
    upsert("evidence.csv", "evidence_id", new_evidence)

    formulations = [
        {"gold_formulation_id":"GF-003","gold_paper_id":"GP-002","formulation_name":"SM-102 eGFP mRNA-LNP","composition_raw":"SM-102:DSPC:cholesterol:DMG-PEG2000 = 50:10:38.5:1.5","composition_basis":"lipid molar ratio","np_ratio":"","formulation_complete":"true","formulation_review_status":"verified_complete","evidence_id":"EVID-012","notes":"Clinically established SM-102 four-lipid formulation."},
        {"gold_formulation_id":"GF-004","gold_paper_id":"GP-005","formulation_name":"LNP1 Onpattro-like Egfp mRNA-LNP","composition_raw":"MC3:DSPC:cholesterol:DMG-PEG2000 = 50:10:38.5:1.5","composition_basis":"lipid molar ratio","np_ratio":"","formulation_complete":"true","formulation_review_status":"verified_complete","evidence_id":"EVID-016","notes":"Structured physicochemical values are frozen from Table 1."},
        {"gold_formulation_id":"GF-005","gold_paper_id":"GP-007","formulation_name":"FITC-HA siMicu1-LNP","composition_raw":"cholesterol; DSPE-PEG; 0.1 mol% FITC-labeled hyaluronic acid; siMicu1","composition_basis":"partial composition","np_ratio":"","formulation_complete":"false","formulation_review_status":"verified_incomplete_ambiguous","evidence_id":"EVID-021","notes":"Missing complete lipid ratios, exact DSPE-PEG identity, and siRNA concentration."},
        {"gold_formulation_id":"GF-006","gold_paper_id":"GP-008","formulation_name":"alpha-CD163/LNP-FAPCAR","composition_raw":"ionizable lipid:DSPC:cholesterol:PEG-lipid = 45:30:23.5:1.5; post-inserted DSPE-PEG-maleimide; anti-CD163","composition_basis":"base lipid molar ratio plus surface conjugation","np_ratio":"","formulation_complete":"false","formulation_review_status":"verified_incomplete_peg_identity","evidence_id":"EVID-025","notes":"Base ratio is complete, but the 1.5 mol% PEG-lipid identity and post-insertion amount are not reported."},
    ]
    upsert("formulations.csv", "gold_formulation_id", formulations)

    component_rows = []
    def component(cid, fid, reported, normalized, role, pct, status, evid, notes=""):
        component_rows.append({"gold_component_id":cid,"gold_formulation_id":fid,"component_name_reported":reported,"component_name_normalized":normalized,"component_role":role,"inchikey":"","molar_percentage":pct,"percentage_unit":"mol_percent" if pct else "","identity_status":status,"identity_source":"article_or_supplement","evidence_id":evid,"notes":notes})
    for cid, name, role, pct in [("GC-009","SM-102","ionizable_lipid","50"),("GC-010","DSPC","helper_lipid","10"),("GC-011","cholesterol","sterol","38.5"),("GC-012","DMG-PEG2000","peg_lipid","1.5")]: component(cid,"GF-003",name,name,role,pct,"identified","EVID-012")
    for cid, name, role, pct in [("GC-013","MC3","ionizable_lipid","50"),("GC-014","DSPC","helper_lipid","10"),("GC-015","cholesterol","sterol","38.5"),("GC-016","DMG-PEG2000","peg_lipid","1.5")]: component(cid,"GF-004",name,name,role,pct,"identified","EVID-016")
    component("GC-017","GF-005","cholesterol","cholesterol","sterol","","identified","EVID-021")
    component("GC-018","GF-005","DSPE-PEG","DSPE-PEG","peg_lipid","","class_only","EVID-021","PEG length not reported.")
    component("GC-019","GF-005","FITC-labeled HA","FITC-hyaluronic acid","targeting_or_tracer_polymer","0.1","class_only","EVID-021")
    component("GC-020","GF-005","siMicu1","Micu1 siRNA","payload","","sequence_not_reported","EVID-021")
    ion="heptadecan-9-yl 8-((2-hydroxyethyl)(8-(nonyloxy)-8-oxooctyl)amino)octanoate"
    component("GC-021","GF-006",ion,ion,"ionizable_lipid","45","identified","EVID-025")
    component("GC-022","GF-006","DSPC","DSPC","helper_lipid","30","identified","EVID-025")
    component("GC-023","GF-006","cholesterol","cholesterol","sterol","23.5","identified","EVID-025")
    component("GC-024","GF-006","PEG-lipid","PEG-lipid","peg_lipid","1.5","class_only","EVID-025","Exact base PEG-lipid identity not reported.")
    component("GC-025","GF-006","DSPE-PEG-maleimide","DSPE-PEG-maleimide","targeting_anchor","","identified","EVID-026","Post-insertion amount not reported.")
    component("GC-026","GF-006","rat anti-mouse CD163","anti-CD163 antibody","targeting_ligand","","biologic_identified","EVID-026","Antibody:LNP ratio reported as 1:20.")
    upsert("components.csv", "gold_component_id", component_rows)

    experiment_fields = ["gold_experiment_id","gold_paper_id","gold_formulation_id","cell_type","delivery_recipient_cell","therapeutic_target_cell","cell_source","species","in_vitro_in_vivo","payload_type","payload_name","reporter","dose","dose_unit","route","timepoint","timepoint_unit","assay","comparator_type","comparator_description","evidence_id","notes"]
    _, existing_experiments = read_csv("experiments.csv")
    role_defaults = {"GX-001":("kupffer_cell",""),"GX-002":("lsec",""),"GX-003":("lsec","lsec")}
    for item in existing_experiments:
        recipient, target = role_defaults.get(item["gold_experiment_id"], (item.get("cell_type", ""), ""))
        item["delivery_recipient_cell"] = recipient
        item["therapeutic_target_cell"] = target
    write_csv("experiments.csv", experiment_fields, existing_experiments)
    experiments = [
        {"gold_experiment_id":"GX-004","gold_paper_id":"GP-002","gold_formulation_id":"GF-003","cell_type":"hepatocyte","delivery_recipient_cell":"hepatocyte","therapeutic_target_cell":"","cell_source":"healthy mouse liver","species":"Mus musculus","in_vitro_in_vivo":"in_vivo","payload_type":"mRNA","payload_name":"eGFP mRNA","reporter":"eGFP","dose":"10","dose_unit":"ug_mRNA_per_mouse","route":"intravenous_tail_vein","timepoint":"24","timepoint_unit":"hour","assay":"fluorescence_imaging_western_blot_IHC","comparator_type":"uninjected_control","comparator_description":"Uninjected healthy mice","evidence_id":"EVID-013","notes":"Direct hepatocyte functional-expression experiment."},
        {"gold_experiment_id":"GX-005","gold_paper_id":"GP-005","gold_formulation_id":"GF-004","cell_type":"kupffer_cell","delivery_recipient_cell":"kupffer_cell","therapeutic_target_cell":"","cell_source":"mouse liver","species":"Mus musculus","in_vitro_in_vivo":"in_vivo","payload_type":"mRNA","payload_name":"Egfp mRNA","reporter":"EGFP","dose":"2","dose_unit":"mg_per_kg","route":"intravenous","timepoint":"16","timepoint_unit":"hour","assay":"intravital_imaging_and_F4_80_staining","comparator_type":"hepatocyte_cell_comparator","comparator_description":"Hepatocyte uptake and translation","evidence_id":"EVID-018","notes":"Designed to distinguish uptake from functional translation."},
        {"gold_experiment_id":"GX-007","gold_paper_id":"GP-007","gold_formulation_id":"GF-005","cell_type":"lsec","delivery_recipient_cell":"liver_unspecified","therapeutic_target_cell":"lsec","cell_source":"mouse liver in HIRI model","species":"Mus musculus","in_vitro_in_vivo":"in_vivo","payload_type":"siRNA","payload_name":"siMicu1","reporter":"FITC tracer","dose":"5","dose_unit":"uL_per_g_formulation_volume","route":"intravenous_tail_vein","timepoint":"2","timepoint_unit":"hour_before_HIRI","assay":"TEM_FITC_FSA_biochemistry_histology","comparator_type":"blank_LNP_and_HIRI_controls","comparator_description":"Blank LNP, HIRI, and siMicu1-LNP plus acteoside arms","evidence_id":"EVID-023","notes":"Direct LSEC outcomes are reported, but cell-specific LNP uptake and siRNA mass dose are not established."},
        {"gold_experiment_id":"GX-008","gold_paper_id":"GP-008","gold_formulation_id":"GF-006","cell_type":"hsc","delivery_recipient_cell":"CD163_positive_macrophage","therapeutic_target_cell":"FAP_positive_activated_HSC","cell_source":"mouse BMDM and activated JS-1 HSC model","species":"Mus musculus","in_vitro_in_vivo":"in_vitro","payload_type":"mRNA","payload_name":"FAPCAR mRNA","reporter":"GFP_or_FAPCAR","dose":"","dose_unit":"","route":"cell_incubation","timepoint":"48","timepoint_unit":"hour","assay":"flow_cytometry_phagocytosis_cytotoxicity","comparator_type":"unmodified_LNP","comparator_description":"Unmodified LNP-FAPCAR and free mRNA","evidence_id":"EVID-028","notes":"Macrophages receive the mRNA; HSCs are target cells for CAR-mediated killing."},
        {"gold_experiment_id":"GX-009","gold_paper_id":"GP-008","gold_formulation_id":"GF-006","cell_type":"hsc","delivery_recipient_cell":"CD163_positive_hepatic_macrophage","therapeutic_target_cell":"FAP_positive_activated_HSC","cell_source":"fibrotic mouse liver","species":"Mus musculus","in_vitro_in_vivo":"in_vivo","payload_type":"mRNA","payload_name":"FAPCAR mRNA","reporter":"FAPCAR","dose":"0.4","dose_unit":"mg_mRNA_per_kg","route":"intravenous_tail_vein","timepoint":"14","timepoint_unit":"day","assay":"histology_immunofluorescence_single_cell_RNAseq","comparator_type":"fibrosis_and_LNP_controls","comparator_description":"Untreated fibrosis, unmodified LNP, and control cargo arms","evidence_id":"EVID-029","notes":"Eligible for HSC therapeutic-effect retrieval, not HSC delivery retrieval."},
    ]
    upsert("experiments.csv", "gold_experiment_id", experiments, experiment_fields)

    outcomes = [
        {"gold_outcome_id":"GO-008","gold_experiment_id":"GX-004","endpoint_family":"functional_expression","endpoint_name":"hepatocyte_eGFP_expression","outcome_value":"","outcome_unit":"","normalization_basis":"HNF4alpha_morphology_identified_hepatocytes","uncertainty_value":"","uncertainty_type":"","qualitative_outcome":"Strong eGFP staining in virtually all hepatocytes.","value_status":"qualitative_reported","evidence_id":"EVID-014","notes":"Healthy liver at 24 hours."},
        {"gold_outcome_id":"GO-010","gold_experiment_id":"GX-005","endpoint_family":"uptake","endpoint_name":"rapid_Kupffer_cell_LNP_uptake","outcome_value":"","outcome_unit":"","normalization_basis":"F4_80_positive_Kupffer_cells","uncertainty_value":"","uncertainty_type":"","qualitative_outcome":"High and rapid uptake comparable to hepatocytes.","value_status":"qualitative_reported","evidence_id":"EVID-019","notes":"Uptake must not be interpreted as protein expression."},
        {"gold_outcome_id":"GO-011","gold_experiment_id":"GX-005","endpoint_family":"functional_expression","endpoint_name":"EGFP_expression_in_Kupffer_cells","outcome_value":"0","outcome_unit":"detectable_cells","normalization_basis":"F4_80_positive_Kupffer_cells_by_intravital_imaging","uncertainty_value":"","uncertainty_type":"","qualitative_outcome":"No obvious EGFP-positive Kupffer cells; expression was observed solely in hepatocytes.","value_status":"reported_below_visual_detection","evidence_id":"EVID-020","notes":"Negative functional-delivery result despite positive uptake."},
        {"gold_outcome_id":"GO-013","gold_experiment_id":"GX-007","endpoint_family":"therapeutic_effect","endpoint_name":"LSEC_ultrastructure_and_function","outcome_value":"","outcome_unit":"","normalization_basis":"HIRI_plus_blank_LNP_control","uncertainty_value":"","uncertainty_type":"","qualitative_outcome":"Reduced mitochondrial damage, calcium accumulation, and LSEC defenestration.","value_status":"qualitative_reported","evidence_id":"EVID-024","notes":"The combination with acteoside produced stronger effects."},
        {"gold_outcome_id":"GO-015","gold_experiment_id":"GX-008","endpoint_family":"functional_expression","endpoint_name":"FAPCAR_or_GFP_positive_BMDMs","outcome_value":"80","outcome_unit":"percent_greater_than","normalization_basis":"BMDMs_exposed_to_alpha_CD163_LNP","uncertainty_value":"","uncertainty_type":"","qualitative_outcome":"Over 80% of BMDMs expressed GFP or FAPCAR.","value_status":"reported_threshold","evidence_id":"EVID-028","notes":"Recipient-cell outcome is macrophage, not HSC."},
        {"gold_outcome_id":"GO-016","gold_experiment_id":"GX-008","endpoint_family":"functional_expression","endpoint_name":"expression_after_unmodified_LNP","outcome_value":"20","outcome_unit":"percent_less_than","normalization_basis":"BMDMs_exposed_to_unmodified_LNP","uncertainty_value":"","uncertainty_type":"","qualitative_outcome":"Unmodified LNP delivered mRNA to fewer than 20% of BMDMs.","value_status":"reported_threshold","evidence_id":"EVID-028","notes":"Comparator for CD163 targeting."},
        {"gold_outcome_id":"GO-017","gold_experiment_id":"GX-009","endpoint_family":"therapeutic_effect","endpoint_name":"activated_HSC_elimination","outcome_value":"","outcome_unit":"","normalization_basis":"FAP_positive_activated_HSCs","uncertainty_value":"","uncertainty_type":"","qualitative_outcome":"FAPCAR macrophages recognized, phagocytosed, and eliminated activated HSC models.","value_status":"qualitative_reported","evidence_id":"EVID-029","notes":"Direct HSC effect without direct HSC LNP delivery."},
        {"gold_outcome_id":"GO-018","gold_experiment_id":"GX-009","endpoint_family":"cell_type_selectivity","endpoint_name":"reporter_recipient_cell","outcome_value":"","outcome_unit":"","normalization_basis":"liver_cell_marker_colocalization","uncertainty_value":"","uncertainty_type":"","qualitative_outcome":"Reporter and FAPCAR expression localized to macrophages rather than Desmin-positive HSCs.","value_status":"qualitative_reported","evidence_id":"EVID-030","notes":"Frozen from an image-based supplementary figure."},
    ]
    upsert("outcomes.csv", "gold_outcome_id", outcomes)

    issues = [
        {"issue_id":"ISS-006","gold_paper_id":"GP-003","entity_type":"paper","entity_id":"GP-003","field_name":"original_experiment_status","issue_type":"review_article","reported_text":"State-of-the-art overview of RNA-targeted therapeutics.","resolution":"Expected answer is zero eligible original formulation-experiment records.","training_eligible":"false","reviewer_notes":"Useful negative paper-level gold case."},
        {"issue_id":"ISS-007","gold_paper_id":"GP-005","entity_type":"outcome","entity_id":"GO-011","field_name":"uptake_versus_expression","issue_type":"endpoint_discordance","reported_text":"High Kupffer uptake with no obvious EGFP translation.","resolution":"Store uptake and functional expression as separate outcomes.","training_eligible":"true","reviewer_notes":"Prevents uptake from being mislabeled as successful protein expression."},
        {"issue_id":"ISS-008","gold_paper_id":"GP-007","entity_type":"formulation","entity_id":"GF-005","field_name":"composition","issue_type":"incomplete_ambiguous_chemistry","reported_text":"Cholesterol, DSPE-PEG, and 0.1 mol% FITC-HA are reported without a complete lipid ratio.","resolution":"Preserve known components and leave missing identities and ratios blank.","training_eligible":"false","reviewer_notes":"Do not use as a complete COMET formulation."},
        {"issue_id":"ISS-009","gold_paper_id":"GP-007","entity_type":"experiment","entity_id":"GX-007","field_name":"dose","issue_type":"volume_without_payload_concentration","reported_text":"5 microliters per gram formulation volume.","resolution":"Record the administered volume and do not infer siRNA mass dose.","training_eligible":"false","reviewer_notes":"Direct LSEC therapeutic evidence remains literature-eligible."},
        {"issue_id":"ISS-010","gold_paper_id":"GP-008","entity_type":"experiment","entity_id":"GX-009","field_name":"cell_role","issue_type":"delivery_recipient_vs_therapeutic_target","reported_text":"CD163-positive macrophages receive FAPCAR mRNA and eliminate FAP-positive HSCs.","resolution":"Store macrophage as delivery_recipient_cell and HSC as therapeutic_target_cell; exclude from HSC-delivery results.","training_eligible":"false","reviewer_notes":"Eligible only for HSC therapeutic-effect retrieval."},
        {"issue_id":"ISS-011","gold_paper_id":"GP-008","entity_type":"formulation","entity_id":"GF-006","field_name":"peg_chemistry","issue_type":"ambiguous_chemistry","reported_text":"The base formulation contains 1.5 mol% PEG-lipid and later receives DSPE-PEG-maleimide post-insertion.","resolution":"Do not assume the base PEG-lipid identity or the post-insertion molar amount.","training_eligible":"false","reviewer_notes":"Ratio-complete but chemically incomplete formulation."},
        {"issue_id":"ISS-012","gold_paper_id":"GP-009","entity_type":"paper","entity_id":"GP-009","field_name":"target_cell_identity","issue_type":"irrelevant_acronym_hit","reported_text":"HSC means hematopoietic stem cell.","resolution":"Exclude from the hepatic stellate cell paper-cell task.","training_eligible":"false","reviewer_notes":"Expected zero eligible hepatic-stellate-cell records."},
    ]
    upsert("issues.csv", "issue_id", issues)

    print("Completed Day 4 gold annotations:")
    for name in ("papers.csv","formulations.csv","components.csv","experiments.csv","outcomes.csv","evidence.csv","issues.csv"):
        _, rows = read_csv(name)
        print(f"  {name}: {len(rows)} rows")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites tracked files in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    main()


if __name__ == "__main__":
    main()
