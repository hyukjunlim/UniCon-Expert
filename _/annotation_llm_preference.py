import re
import os
import random
import time
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from chemprop.data import get_data, MoleculeDataLoader
from chemprop.features import set_reaction, set_explicit_h, set_adding_hs, set_keeping_atom_map
from google import genai
from google.genai import types
from dotenv import load_dotenv

from eval_attention import (
    _get_baseline_info,
    _get_gt_info,
    _load_baseline_models,
    _setup_args,
)
from eval_template_baseline import get_condition_labels, load_libraries
from utils import load_unified_vocab

load_dotenv(override=True)

# --- RESEARCH CONFIGURATION ---

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-3.1-flash-lite-preview"
# MODEL_ID = "gemma-4-31b-it"
INPUT_CSV = "annotation_input_data.csv"
OUTPUT_ANNOTATIONS_LLM = "annotation_llm.csv"
OUTPUT_DIR_BASE = "figs/llm_preference"
SAMPLES_TO_AUDIT = 50 # Number of reaction indices to process
LABEL_PATH = "./data/labels"
LIBRARY_PATH = "./data/condition_library"
LLM_MAX_RETRIES = 6
LLM_INITIAL_BACKOFF_SEC = 5
LLM_MAX_BACKOFF_SEC = 90
LLM_REQUEST_DELAY_SEC = 1.5
OUTPUT_ANNOTATIONS_LLM_JSON = "annotation_llm.json"

# Options: 'Ours', 'GT', 'Baseline'
ALLY = "GT"
OPPONENT = "Baseline" 

OUTPUT_DIR = f"{OUTPUT_DIR_BASE}/{ALLY}_vs_{OPPONENT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Client
client = genai.Client(api_key=API_KEY)

class ChemicalAuditPipeline:
    def __init__(self):
        self.results_data = []

    def _load_existing_results(self):
        """Loads previous audit results so interrupted runs can resume."""
        audit_path = f"{OUTPUT_DIR}/audit_data.csv"
        if not os.path.exists(audit_path):
            return set(), set(), False

        df = pd.read_csv(audit_path)
        failed_mask = df["winner_model"].eq("Failed") if "winner_model" in df else pd.Series(False, index=df.index)
        df_completed = df[~failed_mask].copy()
        self.results_data = df_completed.to_dict("records")
        has_audit_index = "audit_index" in df
        completed_indices = set(df_completed["audit_index"].dropna().astype(int)) if has_audit_index else set()
        completed_reactions = set(df_completed["reaction_smiles"].astype(str)) if "reaction_smiles" in df else set()
        print(
            f"[RESUME] Loaded {len(df_completed)} completed audit results from {audit_path}"
            f" ({int(failed_mask.sum())} failed rows will be retried)."
        )
        return completed_indices, completed_reactions, has_audit_index

    def _results_dataframe(self):
        """Returns audit results sorted by their original sample index."""
        df = pd.DataFrame(self.results_data)
        if "audit_index" in df:
            df = df.assign(audit_index=pd.to_numeric(df["audit_index"], errors="coerce"))
            df = df.sort_values("audit_index", kind="stable", na_position="last").reset_index(drop=True)
        return df

    def generate_template_baseline_input_csv(self):
        """Generates the LLM audit input CSV from GT and template-baseline predictions."""
        print(f"[STAGE 1] Generating template-baseline predictions for {SAMPLES_TO_AUDIT} samples...")
        try:
            args = _setup_args(SAMPLES_TO_AUDIT)
            set_reaction(args.reaction, args.reaction_mode)
            set_explicit_h(args.explicit_h)
            set_adding_hs(args.adding_h)
            set_keeping_atom_map(args.keeping_atom_map)

            total_vocab_size, vocab_mappings, col_to_file, _, idx_to_entity = load_unified_vocab(
                args, logger=None, allow_load=True, use_dmpnn=getattr(args, 'use_dmpnn_reagents', False)
            )

            condition_key = get_condition_labels(LABEL_PATH)
            libs = load_libraries(LIBRARY_PATH)
            baseline_models, _ = _load_baseline_models(args, col_to_file, vocab_mappings, idx_to_entity)

            data_path = args.separate_test_path if (
                args.separate_test_path and os.path.exists(args.separate_test_path)
            ) else args.data_path

            if not os.path.exists(data_path):
                print(f"Error: Data file not found at {data_path}")
                return False

            df_data = pd.read_csv(data_path)
            data = get_data(path=data_path, args=args)
            loader = MoleculeDataLoader(dataset=data, batch_size=1, num_workers=0, shuffle=False)

            rows = []
            for count, batch in enumerate(loader):
                if count >= args.max_data_size:
                    break

                graph_input = batch.batch_graph()
                targets = batch.targets()
                smiles = batch.smiles()[0][0]
                templates = (
                    df_data['tpl_SMARTS_r1'].iloc[count],
                    df_data['tpl_SMARTS_r0'].iloc[count],
                    df_data['tpl_SMARTS_r0*'].iloc[count],
                )

                gt_str, _ = _get_gt_info(targets, args, col_to_file, vocab_mappings, idx_to_entity)
                baseline_strs, _ = _get_baseline_info(
                    baseline_models, graph_input, templates, libs, condition_key,
                    col_to_file, vocab_mappings, idx_to_entity
                )
                baseline_str = baseline_strs[0] if baseline_strs else "None"

                rows.append({
                    "reaction_smiles": smiles,
                    "condition_a": gt_str if gt_str else "None",
                    "condition_b": baseline_str if baseline_str else "None",
                })
                print(f"  Sample {count}: GT={rows[-1]['condition_a']} | Baseline={rows[-1]['condition_b']}")

            pd.DataFrame(rows).to_csv(INPUT_CSV, index=False)
            print(f"Template-baseline input saved to {INPUT_CSV}")
            return True
        except Exception as e:
            print(f"!! Error during template-baseline prediction: {e}")
            return False

    def run_viz_script(self):
        """Compatibility wrapper for the old DETR generation stage."""
        return self.generate_template_baseline_input_csv()

    def load_csv_data(self):
        """Loads samples from the generated CSV file."""
        if not os.path.exists(INPUT_CSV):
            print(f"Input file {INPUT_CSV} not found.")
            return []
        
        df = pd.read_csv(INPUT_CSV)
        parsed_results = []
        for _, row in df.iterrows():
            rxn = row['reaction_smiles']
            labels = {
                "GT": row['condition_a'],
                "Baseline": row['condition_b']
            }
            parsed_results.append((rxn, labels))
        return parsed_results

    def evaluate_with_llm(self, reaction, ally_val, opp_val):
        """
        Checks for exact matches before calling the LLM.
        """
        # Standardize strings
        a_val = str(ally_val).strip()
        o_val = str(opp_val).strip()

        # 1. Exact Match Logic (Ally vs Opponent)
        if a_val == o_val and a_val != "None" and a_val != "":
            return {
                "reaction_smiles": reaction,
                "shown_option_1": a_val,
                "shown_option_2": o_val,
                "user_choice": "Tie",
                "is_option_1_GT": True,
                "winner_model": "Match",
                "confidence": 5,
                "rationale": f"{ALLY} prediction is identical to the {OPPONENT}.",
                "llm_decision_json": json.dumps({
                    "decision": "Tie",
                    "reasoning": f"{ALLY} prediction is identical to the {OPPONENT}.",
                    "confidence": 5,
                }, ensure_ascii=False),
                "raw_response": "",
            }
            
        # Handle the case where both are None
        if (a_val == "None" or a_val == "" or a_val.lower() == "nan") and (o_val == "None" or o_val == "" or o_val.lower() == "nan"):
            return {
                "reaction_smiles": reaction,
                "shown_option_1": "None",
                "shown_option_2": "None",
                "user_choice": "Tie",
                "is_option_1_GT": True,
                "winner_model": "None",
                "confidence": 5,
                "rationale": f"Both {ALLY} and {OPPONENT} predicted no reagents.",
                "llm_decision_json": json.dumps({
                    "decision": "Tie",
                    "reasoning": f"Both {ALLY} and {OPPONENT} predicted no reagents.",
                    "confidence": 5,
                }, ensure_ascii=False),
                "raw_response": "",
            }

        # Proceed to blinded LLM evaluation
        return self.run_blinded_llm_comparison(reaction, a_val, o_val)

    def _build_llm_prompt(self, reaction, prompt_options, valid_labels_str):
        prompt = f"""
You are an impartial senior research chemist adjudicator.

Task:
Compare two anonymized condition predictions for the same chemical reaction.

Reaction:
{reaction}

{prompt_options}

Evaluation criteria:
1. Chemical plausibility for the given reaction.
2. Appropriateness of reagents, catalysts, solvents, bases, acids, oxidants, reductants, and temperature if provided.
3. Compatibility with the reaction transformation.
4. Penalize chemically irrelevant, impossible, or unsupported conditions.
5. Do not prefer an option merely because it is listed first.
6. Do not prefer longer conditions unless the extra components are chemically justified.
7. Treat "None" as a prediction that the reaction proceeds without additional specified conditions.

Decision labels:
- Option 1: choose if Option 1 is clearly more chemically reasonable.
- Option 2: choose if Option 2 is clearly more chemically reasonable.
- Tie: choose if both options are chemically reasonable and roughly equivalent.
- Bad: choose if both options are chemically unreasonable, or the reaction itself is unreasonable.

Return valid JSON only.
Do not wrap the JSON in Markdown.

JSON schema:
{{
"decision": "{valid_labels_str} / Tie / Bad",
"reasoning": "One sentence mechanistic explanation and one sentence decision rationale.",
"confidence": "integer from 1 to 5"
}}

Important:
The confidence must be an integer from 1 to 5.
Use 5 only when the decision is chemically obvious.
Use 3 when the decision is plausible but not certain.
Use 1 or 2 when the comparison is ambiguous.
"""
        return prompt

    def _parse_llm_json(self, res_text):
        """Parses JSON-mode responses, with a small fallback for fenced JSON."""
        if not res_text:
            raise ValueError("empty LLM response")

        text = res_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not json_match:
                raise
            parsed = json.loads(json_match.group(0))

        decision = str(parsed.get("decision", "")).strip()
        decision = re.sub(r"\s+", " ", decision)
        decision = decision.replace("Option1", "Option 1").replace("Option2", "Option 2")
        if decision not in {"Option 1", "Option 2", "Tie", "Bad"}:
            raise ValueError(f"invalid decision: {decision!r}")

        confidence = int(parsed.get("confidence", 0))
        if confidence < 1 or confidence > 5:
            raise ValueError(f"invalid confidence: {confidence!r}")

        reasoning = str(parsed.get("reasoning", "")).strip()
        if not reasoning:
            raise ValueError("missing reasoning")

        return {
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence,
        }

    def _make_failed_result(self, reaction, option_map, error, raw_response=None):
        return {
            "reaction_smiles": reaction,
            "shown_option_1": option_map.get("Option 1", "None"),
            "shown_option_2": option_map.get("Option 2", "None"),
            "user_choice": "Failed",
            "is_option_1_GT": False,
            "winner_model": "Failed",
            "confidence": 0,
            "rationale": str(error),
            "llm_decision_json": "",
            "raw_response": raw_response or "",
        }

    def _prepare_blinded_options(self, candidates):
        # value -> list of identities
        val_to_identities = {}
        for identity, val in candidates.items():
            if val not in val_to_identities:
                val_to_identities[val] = []
            val_to_identities[val].append(identity)
            
        # Create list of unique options
        unique_values = list(val_to_identities.keys())
        random.shuffle(unique_values)
        
        # Build Prompt mapping
        labels = ["Option 1", "Option 2"]
        prompt_options = ""
        option_map = {} # "Option 1" -> value string
        
        for i, val in enumerate(unique_values):
            label = labels[i]
            prompt_options += f"{label}: {val}\n        "
            option_map[label] = val

        valid_labels_str = " / ".join(labels[:len(unique_values)])
        
        return prompt_options, valid_labels_str, option_map, val_to_identities

    def run_blinded_llm_comparison(self, reaction, ally_val, opp_val):
        """Adjudicates conflicts using a randomized, blinded LLM prompt."""
        
        candidates = {
            OPPONENT: opp_val,
            ALLY: ally_val
        }
            
        prompt_options, valid_labels_str, option_map, val_to_identities = self._prepare_blinded_options(candidates)
        prompt = self._build_llm_prompt(reaction, prompt_options, valid_labels_str)

        parsed = None
        last_error = None
        last_response_text = None
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0, 
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_level="minimal")
                    )
                )
                last_response_text = response.text
                parsed = self._parse_llm_json(last_response_text)
                break
            except Exception as e:
                last_error = e
                error_text = str(e)
                transient = isinstance(e, (json.JSONDecodeError, ValueError)) or any(
                    token in error_text
                    for token in (
                        "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                        "deadline", "timeout"
                    )
                )
                if not transient or attempt == LLM_MAX_RETRIES:
                    print(f"LLM Error after {attempt}/{LLM_MAX_RETRIES} attempts: {e}")
                    return self._make_failed_result(reaction, option_map, e, last_response_text)

                backoff = min(LLM_INITIAL_BACKOFF_SEC * (2 ** (attempt - 1)), LLM_MAX_BACKOFF_SEC)
                backoff += random.uniform(0, 2)
                print(
                    f"LLM retryable error ({attempt}/{LLM_MAX_RETRIES}): {e}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)

        else:
            return self._make_failed_result(
                reaction,
                option_map,
                last_error or "LLM retries exhausted",
                last_response_text,
            )

        # Unblind the winner for internal stats
        winner_raw = parsed["decision"]
        
        winner_identity = "Unknown"
        if winner_raw == "Tie":
            winner_identity = "Tie"
        elif winner_raw == "Bad":
            winner_identity = "Bad"
        else:
            winning_val = option_map.get(winner_raw)
            if winning_val is not None:
                winning_identities = val_to_identities.get(winning_val, [])
                if OPPONENT in winning_identities:
                    winner_identity = OPPONENT
                elif ALLY in winning_identities:
                    winner_identity = ALLY

        return {
            "reaction_smiles": reaction,
            "shown_option_1": option_map.get("Option 1", "None"),
            "shown_option_2": option_map.get("Option 2", "None"),
            "user_choice": winner_raw,
            "is_option_1_GT": (val_to_identities.get(option_map.get("Option 1"), []) == [ALLY]) if "Option 1" in option_map else False,
            "winner_model": winner_identity,
            "confidence": parsed["confidence"],
            "rationale": parsed["reasoning"],
            "llm_decision_json": json.dumps(parsed, ensure_ascii=False),
            "raw_response": last_response_text or "",
        }

    def _plot_preference_distribution(self, df, palette):
        plt.figure(figsize=(10, 6))
        order = ["Match", ALLY, OPPONENT, "Tie", "Bad", "None", "Failed"]
        # Filter order to only include categories present in the data
        present_order = [o for o in order if o in df['winner_model'].unique()]
        
        sns.countplot(data=df, x='winner_model', hue='winner_model', legend=False, palette=palette, order=present_order)
        plt.title(f"Tiered Outcome Distribution: {ALLY} vs. {OPPONENT}", fontsize=16)
        plt.xlabel("Outcome", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.savefig(f"{OUTPUT_DIR}/fig_preference.png", dpi=300)

    def generate_research_plots(self):
        """Generates publication-ready figures using Seaborn."""
        if not self.results_data: 
            print("No results to plot.")
            return
        
        df = self._results_dataframe()
        
        # Save full audit data for research
        df.to_csv(f"{OUTPUT_DIR}/audit_data.csv", index=False)
        df.to_json(f"{OUTPUT_DIR}/audit_data.json", orient="records", indent=2, force_ascii=False)
        print(f"audit data saved to {OUTPUT_DIR}/audit_data.csv")
        
        # Save specific format for app annotations as requested
        app_cols = ['reaction_smiles', 'shown_option_1', 'shown_option_2', 'user_choice', 'is_option_1_GT']
        df_app = df[app_cols]
        df_app.to_csv(OUTPUT_ANNOTATIONS_LLM, index=False)
        df_app.to_json(OUTPUT_ANNOTATIONS_LLM_JSON, orient="records", indent=2, force_ascii=False)
        print(f"LLM Annotations saved to {OUTPUT_ANNOTATIONS_LLM}")

        sns.set_theme(style="whitegrid")

        # FIGURE A: Preference Distribution
        palette = {
            "Match": "#55A868",        # Green for exact matches
            ALLY: "#4C72B0",          # Blue for ally wins
            OPPONENT: "#C44E52",      # Red for opponent wins
            "Tie": "#8C8C8C",          # Grey for ties
            "Bad": "#000000",          # Black for both bad
            "None": "#CCB974",         # Yellow/Gold for double background
            "Failed": "#8172B3"        # Purple for exhausted LLM/parser failures
        }
        
        self._plot_preference_distribution(df, palette)

    def execute(self):
        success = self.run_viz_script()
        if not success:
            print("Visualization script failed or didn't generate output.")
            return
            
        parsed_samples = self.load_csv_data()
        print(f"[STAGE 2] CSV loading complete. Found {len(parsed_samples)} samples.")
        completed_indices, completed_reactions, has_audit_index = self._load_existing_results()
        
        for i, (rxn, labels) in enumerate(parsed_samples):
            if has_audit_index:
                already_completed = i in completed_indices
            else:
                already_completed = str(rxn) in completed_reactions

            if already_completed:
                print(f"   [AUDIT {i+1}/{len(parsed_samples)}] Skipping completed reaction.")
                continue

            print(f"   [AUDIT {i+1}/{len(parsed_samples)}] Processing Reaction...")
            
            # Select Ally and Opponent values based on config
            ally_val = labels.get(ALLY, "None")
            opp_val = labels.get(OPPONENT, "None")
            
            # Compare
            result = self.evaluate_with_llm(rxn, ally_val, opp_val)
            if result:
                result["audit_index"] = i
                self.results_data.append(result)
                completed_indices.add(i)
                print(f"      Winner: {result['user_choice']}")
                df_checkpoint = self._results_dataframe()
                df_checkpoint.to_csv(f"{OUTPUT_DIR}/audit_data.csv", index=False)
                df_checkpoint.to_json(f"{OUTPUT_DIR}/audit_data.json", orient="records", indent=2, force_ascii=False)

            time.sleep(LLM_REQUEST_DELAY_SEC)

        self.generate_research_plots()
        print(f"\n[DONE] Results stored in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    pipeline = ChemicalAuditPipeline()
    pipeline.execute()
