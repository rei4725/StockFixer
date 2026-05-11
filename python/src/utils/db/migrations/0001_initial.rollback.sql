-- 0001_initial ロールバック: 全テーブルを削除する
-- 警告: このスクリプトを実行するとすべてのデータが失われます

DROP TABLE IF EXISTS system_config;
DROP TABLE IF EXISTS data_quality_log;
DROP TABLE IF EXISTS dd_state;
DROP TABLE IF EXISTS paper_short_positions;
DROP TABLE IF EXISTS order_run_summary;
DROP TABLE IF EXISTS experiment_runs;
DROP TABLE IF EXISTS feature_selection_log;
DROP TABLE IF EXISTS paper_real_diff;
DROP TABLE IF EXISTS shap_values;
DROP TABLE IF EXISTS paper_positions;
DROP TABLE IF EXISTS paper_orders;
DROP TABLE IF EXISTS paper_balance;
DROP TABLE IF EXISTS prediction_accuracy;
DROP TABLE IF EXISTS model_metrics;
DROP TABLE IF EXISTS index_membership_history;
DROP TABLE IF EXISTS market_data_raw;
DROP TABLE IF EXISTS prediction_results;
DROP TABLE IF EXISTS stock_features
