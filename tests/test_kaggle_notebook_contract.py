"""Contract tests for the Kaggle evaluation notebook generator."""
from __future__ import annotations

import re
from pathlib import Path


CREATE_NB = Path("kaggle_kernel/new_kernel/create_nb.py")


def _source() -> str:
    return CREATE_NB.read_text(encoding="utf-8")


def test_kaggle_notebook_rejects_degenerate_probe_outputs() -> None:
    source = _source()
    assert "completion_quality_report" in source
    assert "valid_reduction = (not is_degenerate)" in source
    assert "bypass_outputs_are_valid = bypass_quality_rate == 1.0" in source
    assert "DEGENERATE_OUTPUT" in source


def test_kaggle_notebook_has_benign_control_gate() -> None:
    source = _source()
    assert "benign_control_prompts" in source
    assert "benign_valid_rate" in source
    assert "benign_outputs_are_valid = benign_valid_rate == 1.0" in source
    assert "BENIGN_REGRESSION" in source


def test_kaggle_notebook_omits_raw_probe_responses_from_logs() -> None:
    source = _source()
    assert "raw text omitted from logs" in source
    assert "print(f'BYPASSED: {bypassed_resp" not in source


def test_kaggle_notebook_emits_closed_loop_cycle_log() -> None:
    source = _source()
    assert "build_cycle_log" in source
    assert "select_best_sweep_result" in source
    assert "cycle_log" in source
    assert "next_cycle_adjustments" in source
    assert "converged" in source


def test_kaggle_notebook_runs_second_layer_local_cycle_after_failure() -> None:
    source = _source()
    assert "optimization_cycles" in source
    assert "middle_layer_indices" in source
    assert "layer_indices=cycle_config['layer_indices']" in source
    assert "include_final_norm=cycle_config['include_final_norm']" in source
    assert "'cycle_index': 3" in source


def test_kaggle_notebook_uses_multidirection_svd_not_global_mean_only() -> None:
    source = _source()
    assert "svd_extraction" in source
    assert "winsorize_percentile" in source
    assert "directions =" in source
    assert "multi_directional_ablation" in source
    assert "direction_candidates.mean(dim=(0, 1))" not in source


def test_kaggle_notebook_uses_dataset_scale_contrastive_probe_set() -> None:
    source = _source()
    assert "MAX_TRAIN_PROMPTS = 512" in source
    assert "MIN_EVAL_PROMPTS = 100" in source
    assert "MIN_BENIGN_CONTROLS = 100" in source
    assert "build_splits" in source
    assert "harmful_train" in source
    assert "harmless_train" in source
    assert "harmful_eval" in source
    assert "benign_eval" in source
    assert "prompt_source" in source


def test_kaggle_notebook_uses_partial_dataset_loading_not_gated_all_or_nothing() -> None:
    source = _source()
    assert "allow_partial_sources=True" in source
    assert "min_partial_train=64" in source
    assert "['walledai/MaliciousInstruct', 'walledai/HarmBench', 'walledai/AdvBench']" in source


def test_kaggle_notebook_supports_multiple_hf_secret_names_without_literal_token() -> None:
    source = _source()
    assert "'HF_TOKEN'" in source
    assert "'HUGGINGFACE_TOKEN'" in source
    assert "'HF_READ_TOKEN'" in source
    assert re.search(r"hf_[A-Za-z0-9]{20,}", source) is None


def test_kaggle_notebook_uses_search_subset_before_full_verification() -> None:
    source = _source()
    assert "SEARCH_BASELINE_SCAN_LIMIT = 32" in source
    assert "SEARCH_EVAL_LIMIT = 8" in source
    assert "SEARCH_BENIGN_LIMIT = 4" in source
    assert "CANDIDATE_TRIAGE_REFUSAL_LIMIT = 4" in source
    assert "CANDIDATE_TRIAGE_BENIGN_LIMIT = 2" in source
    assert "PARTIAL_VERIFY_LIMIT = 8" in source
    assert "PARTIAL_VERIFY_MIN_VALID_REDUCTION = 0.25" in source
    assert "FULL_VERIFY_LIMIT = 24" in source
    assert "MAX_SEARCH_SECONDS = 2700" in source
    assert "MAX_FOLLOWUP_CYCLES = 1" in source
    assert "MAX_NO_IMPROVEMENT_CYCLES = 1" in source
    assert "MAX_CANDIDATES = 4" in source
    assert "MAX_GENERATION_EVALUATIONS = 192" in source
    assert "search_eval_scan_prompts = eval_prompts[:min(SEARCH_BASELINE_SCAN_LIMIT, len(eval_prompts))]" in source
    assert "search_eval_prompts = []" in source
    assert "search_benign_prompts = benign_control_prompts[:min(SEARCH_BENIGN_LIMIT, len(benign_control_prompts))]" in source
    assert "final_verify_prompts = eval_prompts[:min(FULL_VERIFY_LIMIT, len(eval_prompts))]" in source
    assert "final_verification_target_refusal_count = PARTIAL_VERIFY_LIMIT" in source
    assert "candidate_eval_prompts = search_eval_prompts[:min(CANDIDATE_TRIAGE_REFUSAL_LIMIT, len(search_eval_prompts))]" in source
    assert "candidate_benign_prompts = search_benign_prompts[:min(CANDIDATE_TRIAGE_BENIGN_LIMIT, len(search_benign_prompts))]" in source
    assert "final_verification" in source


def test_kaggle_notebook_filters_search_to_baseline_refusals() -> None:
    source = _source()
    assert "baseline_refusing_items" in source
    assert "SEARCH_BASELINE_REFUSAL_TARGET = SEARCH_EVAL_LIMIT + PARTIAL_VERIFY_LIMIT" in source
    assert "search_baseline_refusing_items = baseline_refusing_items[:SEARCH_EVAL_LIMIT]" in source
    assert "heldout_baseline_refusing_items = baseline_refusing_items[SEARCH_EVAL_LIMIT:SEARCH_BASELINE_REFUSAL_TARGET]" in source
    assert "SEARCH_MIN_BASELINE_REFUSALS" in source
    assert "Filtered search prompts to" in source
    assert "Found {SEARCH_BASELINE_REFUSAL_TARGET} baseline-refusing probes; stopping baseline scan early" in source
    assert "scanned_prompt_count = len(baseline_scores)" in source
    assert "search_evidence_ready" in source
    assert "skipping expensive search and saving diagnostics" in source


def test_kaggle_notebook_filters_final_verification_to_baseline_refusals() -> None:
    source = _source()
    assert "final_baseline_refusing_items" in source
    assert "seed_final_baseline_refusals = heldout_baseline_refusing_items[:final_verification_target_refusal_count]" in source
    assert "final_scanned_prompt_set = set(item[0] for item in seed_final_baseline_refusals)" in source
    assert "if prompt in final_scanned_prompt_set:" in source
    assert "Filtered final verification to" in source
    assert "'baseline_refusing_count': len(final_baseline_refusing_items)" in source
    assert "Keeping all final verification prompts" not in source
    assert "Skipping final intervention verification" in source
    assert "final_verification_inconclusive" in source
    assert "'baseline_scan_count': int(final_baseline_scan_count)" in source


def test_kaggle_notebook_reports_partial_plateau_without_claiming_success() -> None:
    source = _source()
    assert "partial_signal = bool(" in source
    assert "PARTIAL_SIGNAL_PLATEAU" in source
    assert "if valid_reduction_rate >= PARTIAL_SIGNAL_THRESHOLD:" in source
    assert "status_reasons.append('INSUFFICIENT_VALID_REDUCTION')" in source


def test_kaggle_notebook_logs_strongreject_import_probe() -> None:
    source = _source()
    assert "StrongREJECT import probes" in source
    assert "strong_reject.evaluate" in source


def test_kaggle_notebook_logs_cloned_git_commit() -> None:
    source = _source()
    assert "git -C /kaggle/working/uncensor fetch origin main --depth 1" in source
    assert "git -C /kaggle/working/uncensor checkout --detach FETCH_HEAD" in source
    assert "git -C /kaggle/working/uncensor rev-parse --short HEAD" in source


def test_kaggle_notebook_records_dual_probe_scores_without_raw_outputs() -> None:
    source = _source()
    assert "from src.probes import dual_probe_scores" in source
    assert "'refusal_score': float(" in source
    assert "'harmfulness_score': float(" in source
    assert "print(f'BYPASSED: {bypassed_resp" not in source


def test_kaggle_notebook_records_benchmark_matrix_and_dataset_scale_status() -> None:
    source = _source()
    assert "from src.benchmark_matrix import build_benchmark_matrix" in source
    assert "'benchmark_matrix': benchmark_matrix" in source
    assert "'benchmark_source_metadata': benchmark_source_metadata" in source
    assert "'dataset_scale_verified': bool(benchmark_matrix['dataset_scale'])" in source


def test_kaggle_notebook_computes_benign_kl_for_candidate_ranking() -> None:
    source = _source()
    assert "from src.metrics import kl_divergence_from_logits" in source
    assert "KL_PROMPT_LIMIT = 2" in source
    assert "def last_token_logits(text):" in source
    assert "'benign_kl': float(benign_kl)" in source
    assert "constrained_candidate_score" in source


def test_kaggle_notebook_uses_ranked_layer_window_search() -> None:
    source = _source()
    assert "ranked_layer_indices" in source
    assert "layer_window(" in source
    assert "best_layer_idx" in source
    assert "best_layer_window_radius_1" in source
    assert "top_ranked_layers" in source
    assert "initial_layer_indices = top_ranked_layers" in source


def test_kaggle_notebook_sweeps_direction_subsets_before_expansion() -> None:
    source = _source()
    assert "direction_count" in source
    assert "active_directions = directions[:cycle_config['direction_count']]" in source
    assert "initial_direction_count = 1" in source
    assert "'direction_count': 2" in source
    assert "'layer_strategy': 'top_ranked_layers_wide_direction_2'" in source
    assert "default_direction_count=3" in Path("src/methods.py").read_text(encoding="utf-8")


def test_kaggle_notebook_bounds_followup_cycles_and_stops_on_stagnation() -> None:
    source = _source()
    assert "meaningful_improvement" in source
    assert "should_stop_search" in source
    assert "should_stop_evaluation_budget" in source
    assert "optimization_cycles = optimization_cycles[:MAX_FOLLOWUP_CYCLES]" in source
    assert "stagnant_cycles" in source
    assert "search_stop_reason" in source
    assert "completed_followup_cycles" in source
    assert "search_generation_count" in source
    assert "def search_budget_exhausted():" in source
    assert "print(f'refusal_probe_{prompt_idx}: delta=" not in source
    assert "print(f'refusal_probe_{prompt_idx}')" not in source


def test_kaggle_notebook_uses_second_stage_candidate_grid_and_run_summary() -> None:
    source = _source()
    assert "build_method_search_space" in source
    assert "kaggle_supported_methods" in source
    assert "select_diverse_candidates" in source
    assert "constrained_candidate_score" in source
    assert "build_run_summary" in source
    assert "robust_method_specs = kaggle_supported_methods()" in source
    assert "Robust Kaggle method registry" in source
    assert "candidate_id" in source
    assert "rejected_candidates" in source
    assert "search_summary" in source
    assert "preferred_layer_window_names=('top_ranked_wide', 'best_radius_4', 'best_radius_2')" in source
    assert "preferred_coefficients=(float(best_result.get('coefficient', 0.20)), 0.20, 0.35)" in source
    assert "candidate_search_enabled = search_evidence_ready and search_stop_reason in ('CONTINUE', 'MAX_CYCLES_REACHED')" in source


def test_kaggle_notebook_runs_bounded_partial_final_verification() -> None:
    source = _source()
    assert "partial_final_verification = bool(" in source
    assert "PARTIAL_VERIFY_MIN_VALID_REDUCTION" in source
    assert "final_verification_mode = 'valid_run'" in source
    assert "Running bounded partial final verification" in source
    assert "'mode': final_verification_mode" in source


def test_kaggle_notebook_records_category_metrics_without_raw_probe_logs() -> None:
    source = _source()
    assert "category_metrics" in source
    assert "category_valid_counts" in source
    assert "prompt_categories" in source
    assert "raw text omitted from logs" in source
