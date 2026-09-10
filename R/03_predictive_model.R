# =============================================================================
# Supervised Learning Assignment 4
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Session Information & Reproducibility
# -----------------------------------------------------------------------------
cat("=== Session Information ===\n")
print(sessionInfo())
set.seed(42)

session_info_path <- file.path("Data/output", "session_info.txt")
dir.create(dirname(session_info_path), recursive = TRUE, showWarnings = FALSE)
writeLines(capture.output(sessionInfo()), session_info_path)
cat("Session info saved to:", session_info_path, "\n")

# -----------------------------------------------------------------------------
# 1. Package Management
# -----------------------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  readr,      # Fast CSV reading
  ggplot2,    # Plotting
  dplyr,      # Data manipulation
  tidyr,      # Data tidying
  caret,      # Model training & evaluation
  pROC        # ROC curves and AUC
)
cat("=== Packages Loaded Successfully ===\n\n")

# -----------------------------------------------------------------------------
# 2. Project Path Configuration
# -----------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

MATCHES_FILE <- "charting-m-matches.csv"
STATS_FILE   <- "charting-m-stats-overview.csv"
OUT_FIGS     <- "output/figures"
OUT_TABLES   <- "output/tables"

dir.create(OUT_FIGS,   recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_TABLES, recursive = TRUE, showWarnings = FALSE)

cat("=== Path Configuration ===\n")
cat("Matches File :", MATCHES_FILE, "\n")
cat("Stats File   :", STATS_FILE,   "\n")
cat("Figs         :", OUT_FIGS,     "\n")
cat("Tables       :", OUT_TABLES,   "\n\n")

stopifnot(
  "charting-m-Matches.csv not found — place it in Data/ folder" =
    file.exists(MATCHES_FILE),
  "charting-m-stats-overview.csv not found — place it in Data/ folder" =
    file.exists(STATS_FILE)
)

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
cat("=== Loading Data ===\n")

# col_types = cols(.default = col_character()) reads all columns as character
# first, suppressing readr type-guessing warnings. Two rows in the matches
# file contain junk values ("Zindaras", "Eva Asderaki-Moore") in numeric/
# categorical columns due to charting errors — these are removed in cleaning.
tennis_data_1 <- suppressWarnings(read_csv(
  MATCHES_FILE,
  col_types      = cols(.default = col_character()),
  show_col_types = FALSE
))

# A handful of source rows are ragged (charting artefacts); confirmed they
# are removed by the Section 2 cleaning filters.
parse_issues <- problems(tennis_data_1)
if (nrow(parse_issues) > 0) {
  cat("Note:", nrow(parse_issues), "ragged source rows flagged by readr;",
      "these are dropped during cleaning.\n")
}

tennis_data_2 <- suppressWarnings(
  read_csv(STATS_FILE, show_col_types = FALSE, name_repair = "minimal")
)

cat("Matches file rows :", nrow(tennis_data_1), "\n")
cat("Stats file rows   :", nrow(tennis_data_2), "\n\n")

# =============================================================================
# SECTION 2: MERGE & CLEAN  (consistent with Assignment 2 pipeline)
# =============================================================================
cat("=== Merging Data ===\n")

merged_data <- merge(tennis_data_1, tennis_data_2, by = "match_id")
cat("Merged data rows :", nrow(merged_data), "\n")
cat("Merged data cols :", ncol(merged_data), "\n\n")

cat("=== Cleaning Data ===\n")


merged_data_cleaned <- merged_data %>%
  filter(set == "Total") %>%
  filter(Surface %in% c("Clay", "Grass", "Hard", "Carpet")) %>%
  rename(player1 = `Player 1`) %>%
  mutate(
    Date             = as.Date(as.character(Date), format = "%Y%m%d"),
    serve_pts        = as.numeric(serve_pts),
    aces             = as.numeric(aces),
    dfs              = as.numeric(dfs),
    first_in         = as.numeric(first_in),
    first_won        = as.numeric(first_won),
    second_in        = as.numeric(second_in),
    second_won       = as.numeric(second_won),
    bk_pts           = as.numeric(bk_pts),
    bp_saved         = as.numeric(bp_saved),
    return_pts       = as.numeric(return_pts),
    return_pts_won   = as.numeric(return_pts_won),
    winners_fh       = as.numeric(winners_fh),
    winners_bh       = as.numeric(winners_bh),
    outcome          = ifelse(player == player1, "Win", "Loss"),
    year             = as.integer(substr(as.character(Date), 1, 4)),
    ace_rate         = aces / serve_pts,
    df_rate          = dfs  / serve_pts,
    first_in_pct     = first_in  / serve_pts,
    first_won_pct    = first_won / first_in,
    second_won_pct   = second_won / second_in,
    winners_fh_rate  = winners_fh / serve_pts,
    winners_bh_rate  = winners_bh / serve_pts,
    bp_save_rate     = ifelse(bk_pts > 0, bp_saved / bk_pts, NA)
  ) %>%
  filter(!is.na(first_won_pct), !is.na(second_won_pct)) %>%
  arrange(Date)

cat("Cleaned data rows :", nrow(merged_data_cleaned), "\n")
cat("Cleaned data cols :", ncol(merged_data_cleaned), "\n\n")

cat("=== Verification ===\n")
cat("Outcome distribution:\n"); print(table(merged_data_cleaned$outcome))
cat("\nSurface distribution:\n"); print(table(merged_data_cleaned$Surface))
cat("\nbp_save_rate NAs:", sum(is.na(merged_data_cleaned$bp_save_rate)),
    "(players who faced 0 break points — retained, handled per feature)\n\n")

# =============================================================================
# SECTION 3: FEATURE ENGINEERING  ROLLING PRE-MATCH AVERAGES
# =============================================================================


cat("=== Building Rolling Pre-Match Features ===\n")

FEAT_COLS <- c("ace_rate", "df_rate", "first_in_pct", "first_won_pct",
               "second_won_pct", "winners_fh_rate", "winners_bh_rate",
               "bp_save_rate")
WINDOW    <- 20L
MIN_PAST  <- 5L

# Split by player, sort by date, compute rolling means with no look-ahead
build_rolling <- function(player_df) {
  player_df <- player_df %>% arrange(Date)
  n <- nrow(player_df)
  if (n <= MIN_PAST) return(NULL)
  
  purrr::map_dfr((MIN_PAST + 1L):n, function(i) {
    past <- player_df[max(1L, i - WINDOW):(i - 1L), ]
    row  <- player_df[i, ]
    feat <- colMeans(past[, FEAT_COLS], na.rm = TRUE)
    names(feat) <- paste0("roll_", names(feat))
    data.frame(
      as.list(feat),
      win_rate  = mean(past$outcome == "Win"),
      player    = row$player,
      player1   = row$player1,
      match_id  = row$match_id,
      Date      = row$Date,
      outcome   = row$outcome,
      is_win    = as.integer(row$outcome == "Win"),
      stringsAsFactors = FALSE
    )
  })
}

pre_df <- merged_data_cleaned %>%
  group_by(player) %>%
  group_split() %>%
  lapply(build_rolling) %>%
  dplyr::bind_rows()

cat("Rolling feature rows :", nrow(pre_df), "\n")
cat("Unique matches covered:", n_distinct(pre_df$match_id), "\n")
cat("Win rate (should be ~0.5):", round(mean(pre_df$is_win), 3), "\n")
cat("Missing values in rolling features:",
    sum(is.na(pre_df[, paste0("roll_", FEAT_COLS)])), "\n\n")

# =============================================================================
# SECTION 4: BUILD MATCHUP-LEVEL DATASET (DIFFERENTIAL FEATURES)
# =============================================================================


cat("=== Building Matchup Differentials ===\n")

ROLL_COLS <- c(paste0("roll_", FEAT_COLS), "win_rate")
DIFF_COLS <- paste0("d_", ROLL_COLS)

winners_df <- pre_df %>%
  filter(is_win == 1) %>%
  distinct(match_id, .keep_all = TRUE) %>%
  select(match_id, Date, all_of(ROLL_COLS))

losers_df  <- pre_df %>%
  filter(is_win == 0) %>%
  distinct(match_id, .keep_all = TRUE) %>%
  select(match_id, all_of(ROLL_COLS))

matchup_df <- winners_df %>%
  inner_join(losers_df, by = "match_id", suffix = c("_w", "_l"))

for (col in ROLL_COLS) {
  matchup_df[[paste0("d_", col)]] <-
    matchup_df[[paste0(col, "_w")]] - matchup_df[[paste0(col, "_l")]]
}

# Symmetrise: winner rows (is_win = 1) + mirrored loser rows (is_win = 0)
winner_rows <- matchup_df %>%
  select(all_of(DIFF_COLS), Date) %>%
  mutate(is_win = 1L)

loser_rows  <- matchup_df %>%
  select(all_of(DIFF_COLS), Date) %>%
  mutate(across(all_of(DIFF_COLS), ~ -.), is_win = 0L)

sym_df <- bind_rows(winner_rows, loser_rows) %>%
  arrange(Date)

cat("Matchup dataset rows  :", nrow(sym_df), "\n")
cat("Win rate (should = 0.5):", round(mean(sym_df$is_win), 3), "\n\n")

# =============================================================================
# SECTION 5: TEMPORAL TRAIN / TEST SPLIT
# =============================================================================
# CRITICAL: we use a temporal (chronological) split rather than random.


cat("=== Temporal Train / Test Split ===\n")

cutoff_date <- as.Date(quantile(as.numeric(sym_df$Date), 0.75),
                       origin = "1970-01-01")
cat("Cutoff date (75th percentile):", format(cutoff_date), "\n")

train_set <- sym_df %>% filter(Date <= cutoff_date) %>% select(-Date)
test_set  <- sym_df %>% filter(Date >  cutoff_date) %>% select(-Date)

train_set$is_win <- factor(train_set$is_win, levels = c(0, 1),
                           labels = c("Loss", "Win"))
test_set$is_win  <- factor(test_set$is_win,  levels = c(0, 1),
                           labels = c("Loss", "Win"))

cat("Train rows :", nrow(train_set), "| Win rate:",
    round(mean(train_set$is_win == "Win"), 3), "\n")
cat("Test rows  :", nrow(test_set),  "| Win rate:",
    round(mean(test_set$is_win  == "Win"), 3), "\n\n")

# =============================================================================
# SECTION 6: PREPROCESSING — CENTRE AND SCALE
# =============================================================================

# The preprocessing object is fitted on the TRAINING set only,
# then applied to the test set. Fitting on the full dataset would leak
# test-set distribution information into the model.

cat("=== Preprocessing: Centre and Scale ===\n")

pre_proc <- preProcess(train_set[, DIFF_COLS], method = c("center", "scale"))

train_scaled               <- train_set
test_scaled                <- test_set
train_scaled[, DIFF_COLS]  <- predict(pre_proc, train_set[, DIFF_COLS])
test_scaled[, DIFF_COLS]   <- predict(pre_proc, test_set[, DIFF_COLS])

cat("Preprocessing fitted on training set only (no leakage).\n\n")

# =============================================================================
# SECTION 7: MODEL TRAINING — LOGISTIC REGRESSION WITH 5-FOLD CV
# =============================================================================

# 5-fold stratified cross-validation is performed on the training set to
# estimate generalisation performance before final evaluation on the test set.

cat("=== Training Logistic Regression (5-fold CV) ===\n")

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

logit_model <- train(
  is_win    ~ .,
  data      = train_scaled,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "ROC"
)

cat("5-Fold CV Results (training set):\n")
print(logit_model$results[, c("ROC", "Sens", "Spec")])
cat("\n")

# =============================================================================
# SECTION 8: MODEL EVALUATION ON HELD-OUT TEST SET
# =============================================================================

cat("=== Test Set Evaluation ===\n")

pred_class <- predict(logit_model, newdata = test_scaled)
pred_prob  <- predict(logit_model, newdata = test_scaled,
                      type = "prob")[, "Win"]

# -- Confusion matrix --
cm <- confusionMatrix(pred_class, test_scaled$is_win, positive = "Win")
cat("Confusion Matrix:\n"); print(cm$table); cat("\n")

# -- Classification metrics --
cat("=== Performance Metrics ===\n")
cat(sprintf("  Accuracy   : %.4f\n", cm$overall["Accuracy"]))
cat(sprintf("  Precision  : %.4f\n", cm$byClass["Precision"]))
cat(sprintf("  Recall     : %.4f\n", cm$byClass["Recall"]))
cat(sprintf("  F1 Score   : %.4f\n", cm$byClass["F1"]))
cat(sprintf("  Kappa      : %.4f\n", cm$overall["Kappa"]))

# -- Probability metrics --
roc_obj <- roc(as.integer(test_scaled$is_win == "Win"),
               pred_prob, quiet = TRUE)
brier   <- mean((pred_prob - as.integer(test_scaled$is_win == "Win"))^2)
cat(sprintf("  AUC-ROC    : %.4f\n", auc(roc_obj)))
cat(sprintf("  Brier Score: %.4f  (0 = perfect, 0.25 = no-skill baseline)\n",
            brier))
cat("\n")

# -- Save metrics table --
metrics_df <- data.frame(
  Metric = c("Accuracy","Precision","Recall","F1","Kappa","AUC-ROC","Brier Score"),
  Value  = round(c(cm$overall["Accuracy"], cm$byClass["Precision"],
                   cm$byClass["Recall"],   cm$byClass["F1"],
                   cm$overall["Kappa"],    as.numeric(auc(roc_obj)), brier), 4)
)
write.csv(metrics_df,
          file.path(OUT_TABLES, "model_performance_metrics.csv"),
          row.names = FALSE)
cat("Metrics table saved.\n\n")

# =============================================================================
# SECTION 9: VARIABLE IMPORTANCE & COEFFICIENTS
# =============================================================================

cat("=== Variable Importance ===\n")
vi <- varImp(logit_model)
print(vi)

cat("\n=== Model Coefficients (log-odds scale) ===\n")
coef_tbl <- summary(logit_model$finalModel)$coefficients
print(round(coef_tbl, 4))

# Save coefficients
coef_df <- as.data.frame(round(coef_tbl, 4))
coef_df$Feature <- rownames(coef_df)
write.csv(coef_df,
          file.path(OUT_TABLES, "model_coefficients.csv"),
          row.names = FALSE)
cat("Coefficients table saved.\n\n")

# =============================================================================
# SECTION 10: VISUALISATIONS
# =============================================================================

save_plot <- function(filename, plot_expr, width = 8, height = 5) {
  filepath <- file.path(OUT_FIGS, filename)
  ggsave(filepath, plot = plot_expr, width = width, height = height, dpi = 150)
  cat("Saved:", filepath, "\n")
}

# -- ROC Curve --
roc_df <- data.frame(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)
p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(colour = "steelblue", linewidth = 1) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  labs(
    title    = "ROC Curve — Pre-Match Win Probability Model",
    subtitle = sprintf("AUC = %.4f", as.numeric(auc(roc_obj))),
    x        = "False Positive Rate (1 - Specificity)",
    y        = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal()
save_plot("09_roc_curve.png", p_roc)

# -- Win probability distribution by actual outcome --
prob_df <- data.frame(
  prob    = pred_prob,
  outcome = test_scaled$is_win
)
p_prob <- ggplot(prob_df, aes(x = prob, fill = outcome)) +
  geom_histogram(bins = 40, alpha = 0.7, position = "identity",
                 colour = "white") +
  scale_fill_manual(values = c("Loss" = "tomato", "Win" = "steelblue")) +
  labs(
    title = "Predicted Win Probability by Actual Outcome",
    x     = "Predicted Win Probability",
    y     = "Count",
    fill  = "Actual Outcome"
  ) +
  theme_minimal()
save_plot("10_win_probability_distribution.png", p_prob)

# -- Variable importance bar chart --
vi_df <- data.frame(
  Feature    = rownames(vi$importance),
  Importance = vi$importance[, 1]
) %>% arrange(Importance)
vi_df$Feature <- factor(vi_df$Feature, levels = vi_df$Feature)

p_vi <- ggplot(vi_df, aes(x = Feature, y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Variable Importance — Logistic Regression",
    x     = "Feature",
    y     = "Importance (scaled 0-100)"
  ) +
  theme_minimal()
save_plot("11_variable_importance.png", p_vi)

# =============================================================================
# SECTION 11: DONE
# =============================================================================
cat("\n=== All outputs saved ===\n")
cat("Figures :", OUT_FIGS,   "\n")
cat("Tables  :", OUT_TABLES, "\n\n")
print(sessionInfo())

