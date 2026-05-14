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
    assert "expanded_layer_indices" in source
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
    assert "SEARCH_EVAL_LIMIT = 16" in source
    assert "SEARCH_BENIGN_LIMIT = 16" in source
    assert "FULL_VERIFY_LIMIT = 100" in source
    assert "search_eval_prompts = eval_prompts[:min(SEARCH_EVAL_LIMIT, len(eval_prompts))]" in source
    assert "search_benign_prompts = benign_control_prompts[:min(SEARCH_BENIGN_LIMIT, len(benign_control_prompts))]" in source
    assert "final_verify_prompts = eval_prompts[:min(FULL_VERIFY_LIMIT, len(eval_prompts))]" in source
    assert "final_verification" in source


def test_kaggle_notebook_filters_search_to_baseline_refusals() -> None:
    source = _source()
    assert "baseline_refusing_items" in source
    assert "SEARCH_MIN_BASELINE_REFUSALS" in source
    assert "Filtered search prompts to" in source


def test_kaggle_notebook_filters_final_verification_to_baseline_refusals() -> None:
    source = _source()
    assert "final_baseline_refusing_items" in source
    assert "Filtered final verification to" in source
    assert "'baseline_refusing_count': len(final_baseline_refusing_items)" in source


def test_kaggle_notebook_logs_strongreject_import_probe() -> None:
    source = _source()
    assert "StrongREJECT import probes" in source
    assert "strong_reject.evaluate" in source


def test_kaggle_notebook_logs_cloned_git_commit() -> None:
    source = _source()
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
    assert "KL_PROMPT_LIMIT = 4" in source
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


def test_kaggle_notebook_sweeps_direction_subsets_before_expansion() -> None:
    source = _source()
    assert "direction_count" in source
    assert "active_directions = directions[:cycle_config['direction_count']]" in source
    assert "'direction_count': 1" in source
    assert "'direction_count': 2" in source
    assert "'direction_count': 3" in source


def test_kaggle_notebook_expands_when_quality_is_preserved_but_movement_is_low() -> None:
    source = _source()
    assert "top_ranked_layers_wide" in source
    assert "best_layer_window_radius_4" in source
    assert "'layer_strategy': 'top_ranked_layers_wide_no_regression_expansion'" in source
    assert "'layer_strategy': 'best_layer_window_radius_4_no_regression_expansion'" in source
    assert "'coefficient_grid': [0.20, 0.30, 0.40, 0.50, 0.60]" in source


def test_kaggle_notebook_uses_second_stage_candidate_grid_and_run_summary() -> None:
    source = _source()
    assert "build_intervention_candidates" in source
    assert "constrained_candidate_score" in source
    assert "build_run_summary" in source
    assert "direction_families=['svd_primary', 'svd_multi']" in source
    assert "intervention_types=['hook_ablation']" in source
    assert "candidate_id" in source
    assert "rejected_candidates" in source
    assert "search_summary" in source


def test_kaggle_notebook_records_category_metrics_without_raw_probe_logs() -> None:
    source = _source()
    assert "category_metrics" in source
    assert "category_valid_counts" in source
    assert "prompt_categories" in source
    assert "raw text omitted from logs" in source
