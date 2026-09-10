# =============================================================================
# Assignment 5  Validation & Model Comparison

# =============================================================================

# -----------------------------------------------------------------------------
# 0. Packages & reproducibility
# -----------------------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")

suppressPackageStartupMessages(
  pacman::p_load(
    readr,         # CSV reading
    dplyr,         # data manipulation
    ggplot2,       # plotting
    caret,         # train/test split, CV, preprocessing, metrics
    pROC,          # ROC curves and AUC
    glmnet,        # regularised logistic regression
    randomForest,  # random forest
    cluster,       # silhouette analysis
    e1071          # caret companion (confusionMatrix support)
  )
)

set.seed(42)  # global seed for reproducibility


# -----------------------------------------------------------------------------
# 1. Paths & output folders
# -----------------------------------------------------------------------------
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  doc_path <- rstudioapi::getActiveDocumentContext()$path
  if (nzchar(doc_path)) setwd(dirname(doc_path))
}
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

DATA_DIR     <- "."
MATCHES_FILE <- "charting-m-matches.csv"
STATS_FILE   <- "charting-m-stats-overview.csv"  
matches_path <- file.path(DATA_DIR, MATCHES_FILE)
stats_path   <- file.path(DATA_DIR, STATS_FILE)
# -----------------------------------------------------------------------------
# 2. Load data
# -----------------------------------------------------------------------------

tennis_matches <- suppressWarnings(read_csv(
  matches_path,
  col_types      = cols(.default = col_character()),
  show_col_types  = FALSE
))

tennis_stats <- suppressWarnings(read_csv(
  stats_path,
  show_col_types = FALSE,
  name_repair    = "minimal"
))

cat("Matches rows:", nrow(tennis_matches),
    "| Stats rows:", nrow(tennis_stats), "\n")


# -----------------------------------------------------------------------------
# 3. Merge & clean  (consistent with the Assignment 2 / 4 pipeline)
# -----------------------------------------------------------------------------

merged <- merge(tennis_matches, tennis_stats, by = "match_id")

clean <- merged %>%
  filter(set == "Total") %>%
  filter(Surface %in% c("Clay", "Grass", "Hard", "Carpet")) %>%
  rename(player1 = `Player 1`)


clean <- suppressWarnings(
  clean %>%
    mutate(
      Date           = as.Date(as.character(Date), format = "%Y%m%d"),
      serve_pts      = as.numeric(serve_pts),
      aces           = as.numeric(aces),
      dfs            = as.numeric(dfs),
      first_in       = as.numeric(first_in),
      first_won      = as.numeric(first_won),
      second_in      = as.numeric(second_in),
      second_won     = as.numeric(second_won),
      bk_pts         = as.numeric(bk_pts),
      bp_saved       = as.numeric(bp_saved),
      return_pts     = as.numeric(return_pts),
      return_pts_won = as.numeric(return_pts_won),
      winners        = as.numeric(winners),
      winners_fh     = as.numeric(winners_fh),
      winners_bh     = as.numeric(winners_bh),
      unforced       = as.numeric(unforced),
      outcome        = ifelse(player == player1, "Win", "Loss"),
      year           = as.integer(format(Date, "%Y")),
      ace_rate        = ifelse(serve_pts > 0, aces       / serve_pts, NA_real_),
      df_rate         = ifelse(serve_pts > 0, dfs        / serve_pts, NA_real_),
      first_in_pct    = ifelse(serve_pts > 0, first_in   / serve_pts, NA_real_),
      first_won_pct   = ifelse(first_in  > 0, first_won  / first_in,  NA_real_),
      second_won_pct  = ifelse(second_in > 0, second_won / second_in, NA_real_),
      winners_fh_rate = ifelse(serve_pts > 0, winners_fh / serve_pts, NA_real_),
      winners_bh_rate = ifelse(serve_pts > 0, winners_bh / serve_pts, NA_real_),
      bp_save_rate    = ifelse(bk_pts    > 0, bp_saved   / bk_pts,    NA_real_)
    ) %>%
    filter(!is.na(first_won_pct), !is.na(second_won_pct)) %>%
    arrange(Date)
)

cat("Cleaned rows:", nrow(clean), "\n")
cat("Outcome balance:\n"); print(table(clean$outcome)); cat("\n")


# -----------------------------------------------------------------------------
# 4. Rolling pre-match features (no look-ahead)
# -----------------------------------------------------------------------------

FEAT_COLS <- c("ace_rate", "df_rate", "first_in_pct", "first_won_pct",
               "second_won_pct", "winners_fh_rate", "winners_bh_rate",
               "bp_save_rate")
WINDOW   <- 20L
MIN_PAST <- 5L

build_rolling <- function(player_df) {
  player_df <- player_df[order(player_df$Date), ]
  n <- nrow(player_df)
  if (n <= MIN_PAST) return(NULL)
  
  out <- vector("list", n - MIN_PAST)
  for (i in (MIN_PAST + 1L):n) {
    past <- player_df[max(1L, i - WINDOW):(i - 1L), , drop = FALSE]
    feat <- colMeans(past[, FEAT_COLS, drop = FALSE], na.rm = TRUE)
    names(feat) <- paste0("roll_", names(feat))
    row <- player_df[i, ]
    out[[i - MIN_PAST]] <- data.frame(
      as.list(feat),
      win_rate = mean(past$outcome == "Win"),
      player   = row$player,
      player1  = row$player1,
      match_id = row$match_id,
      Date     = row$Date,
      outcome  = row$outcome,
      is_win   = as.integer(row$outcome == "Win"),
      check.names      = FALSE,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

pre_df <- do.call(rbind, lapply(split(clean, clean$player), build_rolling))
rownames(pre_df) <- NULL

cat("Rolling feature rows:", nrow(pre_df),
    "| Matches covered:", length(unique(pre_df$match_id)), "\n\n")


# -----------------------------------------------------------------------------
# 5. Matchup differentials (Player A minus Player B), symmetrised
# -----------------------------------------------------------------------------

ROLL_COLS <- c(paste0("roll_", FEAT_COLS), "win_rate")
DIFF_COLS <- paste0("d_", ROLL_COLS)

winners_df <- pre_df %>%
  filter(is_win == 1) %>%
  distinct(match_id, .keep_all = TRUE) %>%
  select(match_id, Date, all_of(ROLL_COLS))

losers_df <- pre_df %>%
  filter(is_win == 0) %>%
  distinct(match_id, .keep_all = TRUE) %>%
  select(match_id, all_of(ROLL_COLS))

matchup_df <- inner_join(winners_df, losers_df, by = "match_id",
                         suffix = c("_w", "_l"))

for (col in ROLL_COLS) {
  matchup_df[[paste0("d_", col)]] <-
    matchup_df[[paste0(col, "_w")]] - matchup_df[[paste0(col, "_l")]]
}

winner_rows <- matchup_df %>%
  select(all_of(DIFF_COLS), Date) %>%
  mutate(is_win = 1L)

loser_rows <- matchup_df %>%
  select(all_of(DIFF_COLS), Date) %>%
  mutate(across(all_of(DIFF_COLS), ~ -.), is_win = 0L)

sym_df <- bind_rows(winner_rows, loser_rows) %>% arrange(Date)

# Drop any rows with NA/NaN differentials (e.g. windows with no break points)
n_before <- nrow(sym_df)
sym_df <- sym_df[complete.cases(sym_df[, DIFF_COLS]), ]
cat("Matchup rows:", nrow(sym_df),
    "(dropped", n_before - nrow(sym_df), "incomplete) | Win rate:",
    round(mean(sym_df$is_win), 3), "\n\n")


# -----------------------------------------------------------------------------
# 6. Temporal train / test split
# -----------------------------------------------------------------------------

cutoff_date <- as.Date(quantile(as.numeric(sym_df$Date), 0.75, names = FALSE),
                       origin = "1970-01-01")
cat("Cutoff date (75th percentile):", format(cutoff_date), "\n")

train_set <- subset(sym_df, Date <= cutoff_date, select = -Date)
test_set  <- subset(sym_df, Date >  cutoff_date, select = -Date)

train_set$is_win <- factor(train_set$is_win, levels = c(0, 1),
                           labels = c("Loss", "Win"))
test_set$is_win  <- factor(test_set$is_win,  levels = c(0, 1),
                           labels = c("Loss", "Win"))

cat("Train rows:", nrow(train_set), "| Test rows:", nrow(test_set), "\n\n")


# -----------------------------------------------------------------------------
# 7. Preprocessing — centre & scale (fitted on training data only)
# -----------------------------------------------------------------------------

pre_proc <- preProcess(train_set[, DIFF_COLS], method = c("center", "scale"))

train_scaled <- train_set
test_scaled  <- test_set
train_scaled[, DIFF_COLS] <- predict(pre_proc, train_set[, DIFF_COLS])
test_scaled[,  DIFF_COLS] <- predict(pre_proc, test_set[,  DIFF_COLS])


# -----------------------------------------------------------------------------
# 8. Model training — common 5-fold CV for a fair comparison
# -----------------------------------------------------------------------------

set.seed(42)
folds <- createFolds(train_scaled$is_win, k = 5, returnTrain = TRUE)

ctrl <- trainControl(
  method          = "cv",
  index           = folds,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)


cat("Training logistic regression...\n")
set.seed(42)
logit_model <- suppressWarnings(train(
  is_win ~ ., data = train_scaled,
  method = "glm", family = "binomial",
  trControl = ctrl, metric = "ROC"
))

cat("Training regularised logistic regression (glmnet)...\n")
glmnet_grid <- expand.grid(alpha  = c(0, 0.5, 1),
                           lambda = 10 ^ seq(-4, 0, length.out = 25))
set.seed(42)
glmnet_model <- suppressWarnings(train(
  is_win ~ ., data = train_scaled,
  method = "glmnet",
  trControl = ctrl, metric = "ROC",
  tuneGrid = glmnet_grid
))

cat("Training random forest...\n")
rf_grid <- expand.grid(mtry = c(2, 3, 4))
set.seed(42)
rf_model <- suppressWarnings(train(
  is_win ~ ., data = train_scaled,
  method = "rf",
  trControl = ctrl, metric = "ROC",
  tuneGrid = rf_grid, ntree = 500, importance = TRUE
))

cat("Models trained.\n\n")
cat("Best glmnet tuning  -> alpha:", glmnet_model$bestTune$alpha,
    "| lambda:", round(glmnet_model$bestTune$lambda, 4), "\n")
cat("Best random forest  -> mtry :", rf_model$bestTune$mtry, "\n\n")

glmnet_model$bestTune

# -----------------------------------------------------------------------------
# 9. Evaluation on the held-out test set
# -----------------------------------------------------------------------------

models <- list(
  "Logistic"      = logit_model,
  "Elastic Net"   = glmnet_model,
  "Random Forest" = rf_model
)

results <- lapply(names(models), function(nm) {
  m   <- models[[nm]]
  cls <- predict(m, test_scaled)
  pr  <- predict(m, test_scaled, type = "prob")[, "Win"]
  cm  <- confusionMatrix(cls, test_scaled$is_win, positive = "Win")
  # Explicit levels + direction silence pROC's auto-direction messages.
  roc_obj <- roc(test_scaled$is_win, pr,
                 levels = c("Loss", "Win"), direction = "<", quiet = TRUE)
  list(name = nm, cls = cls, pr = pr, cm = cm, roc = roc_obj)
})
names(results) <- names(models)

# -- Test-set comparison table ------------------------------------------------
comparison <- do.call(rbind, lapply(results, function(r) {
  data.frame(
    Model    = r$name,
    Accuracy = round(unname(r$cm$overall["Accuracy"]), 4),
    AUC      = round(as.numeric(auc(r$roc)), 4),
    F1       = round(unname(r$cm$byClass["F1"]), 4),
    Kappa    = round(unname(r$cm$overall["Kappa"]), 4),
    Brier    = round(mean((r$pr - as.integer(test_scaled$is_win == "Win"))^2), 4),
    stringsAsFactors = FALSE
  )
}))
rownames(comparison) <- NULL

cat("=== Test-Set Model Comparison ===\n")
print(comparison)
write.csv(comparison, file.path(OUT_TABLES, "model_comparison.csv"),
          row.names = FALSE)

# -- Cross-validation ROC (training) vs test, to check for over-fitting -------
cv_summary <- data.frame(
  Model  = c("Logistic", "Elastic Net", "Random Forest"),
  CV_ROC = round(c(max(logit_model$results$ROC),
                   max(glmnet_model$results$ROC),
                   max(rf_model$results$ROC)), 4),
  Test_AUC = comparison$AUC,
  stringsAsFactors = FALSE
)
cat("\n=== Cross-Validated ROC vs Test AUC ===\n")
print(cv_summary)
write.csv(cv_summary, file.path(OUT_TABLES, "cv_vs_test.csv"),
          row.names = FALSE)
cat("\n")


# -----------------------------------------------------------------------------
# 10. Plots & interpretability outputs
# -----------------------------------------------------------------------------
# -- ROC overlay --------------------------------------------------------------
roc_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(Model = r$name,
             fpr = 1 - r$roc$specificities,
             tpr = r$roc$sensitivities,
             stringsAsFactors = FALSE)
}))

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, colour = Model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  labs(title = "ROC Curves — Model Comparison (Held-out Test Set)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)", colour = "Model") +
  theme_minimal()
ggsave(file.path(OUT_FIGS, "01_roc_comparison.png"), p_roc,
       width = 8, height = 5, dpi = 150)

# -- AUC & Accuracy bar chart -------------------------------------------------
metric_long <- data.frame(
  Model  = rep(comparison$Model, 2),
  Metric = rep(c("AUC", "Accuracy"), each = nrow(comparison)),
  Value  = c(comparison$AUC, comparison$Accuracy)
)
p_bar <- ggplot(metric_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  coord_cartesian(ylim = c(0, 1)) +
  labs(title = "Test-Set AUC and Accuracy by Model", x = NULL, y = "Score") +
  theme_minimal()
ggsave(file.path(OUT_FIGS, "02_metric_comparison.png"), p_bar,
       width = 8, height = 5, dpi = 150)

# -- Random forest variable importance ---------------------------------------
rf_imp <- varImp(rf_model)$importance
imp_vals <- if (ncol(rf_imp) > 1) rowMeans(rf_imp) else rf_imp[[1]]
rf_imp_df <- data.frame(Feature = rownames(rf_imp),
                        Importance = imp_vals,
                        stringsAsFactors = FALSE)
rf_imp_df <- rf_imp_df[order(rf_imp_df$Importance), ]
rf_imp_df$Feature <- factor(rf_imp_df$Feature, levels = rf_imp_df$Feature)

p_imp <- ggplot(rf_imp_df, aes(x = Feature, y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Random Forest Variable Importance", x = NULL,
       y = "Importance (0-100)") +
  theme_minimal()
ggsave(file.path(OUT_FIGS, "03_rf_importance.png"), p_imp,
       width = 8, height = 5, dpi = 150)
write.csv(rf_imp_df, file.path(OUT_TABLES, "rf_importance.csv"),
          row.names = FALSE)

# -- Logistic regression coefficients (interpretable log-odds) ----------------
logit_coef <- summary(logit_model$finalModel)$coefficients
logit_coef_df <- data.frame(Feature = rownames(logit_coef),
                            round(logit_coef, 4),
                            row.names = NULL, check.names = FALSE)
write.csv(logit_coef_df, file.path(OUT_TABLES, "logistic_coefficients.csv"),
          row.names = FALSE)

cat("Saved figures to", OUT_FIGS, "and tables to", OUT_TABLES, "\n\n")


# -----------------------------------------------------------------------------
# 11. Unsupervised validation (PCA + k-means) 
# -----------------------------------------------------------------------------
cat("=== Unsupervised validation: PCA + k-means ===\n")


clust_feats <- suppressWarnings(
  clean %>%
    mutate(
      total_points        = serve_pts + return_pts,
      winner_rate         = ifelse(total_points > 0, winners        / total_points, NA_real_),
      unforced_error_rate = ifelse(total_points > 0, unforced       / total_points, NA_real_),
      return_pts_won_pct  = ifelse(return_pts   > 0, return_pts_won / return_pts,   NA_real_)
    ) %>%
    filter(serve_pts <= 465) %>%
    transmute(
      ace_rate, double_fault_rate = df_rate, first_serve_pct = first_in_pct,
      winner_rate, unforced_error_rate, return_pts_won_pct, bp_save_rate
    )
)
clust_feats <- clust_feats[complete.cases(clust_feats), ]

# Drop any zero-variance column so scaling cannot produce NaN (which would
# break prcomp).
keep <- vapply(clust_feats, function(x) sd(x) > 0, logical(1))
clust_feats <- clust_feats[, keep, drop = FALSE]

clust_scaled <- scale(clust_feats)
pca_model <- prcomp(clust_scaled, center = FALSE, scale. = FALSE)
pc2 <- pca_model$x[, 1:2]

cat("Clustering rows:", nrow(pc2),
    "| PC1+PC2 variance explained:",
    round(sum((pca_model$sdev[1:2]^2) / sum(pca_model$sdev^2)), 3), "\n")

# -- Elbow (within-cluster sum of squares), k = 1..10 -------------------------
wss <- vapply(1:10, function(k) {
  set.seed(42)
  kmeans(pc2, centers = k, nstart = 25, iter.max = 300, 
         algorithm = "MacQueen")$tot.withinss
}, numeric(1))

png(file.path(OUT_FIGS, "04_elbow_wss.png"), width = 800, height = 500)
plot(1:10, wss, type = "b", pch = 19,
     xlab = "Number of clusters k", ylab = "Total within-cluster SS",
     main = "K-Means Elbow Method")
dev.off()

# -- Silhouette, k = 2..10 (sampled if large, to bound the distance matrix) ---
set.seed(42)
sil_idx <- if (nrow(pc2) > 5000) sample(seq_len(nrow(pc2)), 5000) else seq_len(nrow(pc2))
pc2_s <- pc2[sil_idx, ]
d <- dist(pc2_s)  # computed once and reused across k

sil <- vapply(2:10, function(k) {
  set.seed(42)
  km <- kmeans(pc2_s, centers = k, nstart = 25, iter.max = 300, algorithm = 
                 "MacQueen")
  mean(silhouette(km$cluster, d)[, 3])
}, numeric(1))

png(file.path(OUT_FIGS, "05_silhouette_scores.png"), width = 800, height = 500)
plot(2:10, sil, type = "b", pch = 19,
     xlab = "Number of clusters k", ylab = "Average silhouette width",
     main = "Silhouette Analysis for Optimal k")
dev.off()

cluster_validation <- data.frame(
  k   = 1:10,
  WSS = round(wss, 2),
  Avg_Silhouette = c(NA, round(sil, 4))  # silhouette undefined for k = 1
)
cat("\nCluster validation summary:\n")
print(cluster_validation)
write.csv(cluster_validation,
          file.path(OUT_TABLES, "cluster_validation.csv"), row.names = FALSE)
cat("\n")

glmnet_model$bestTune
# -----------------------------------------------------------------------------
# 12. Session information (full reproducibility record)
# -----------------------------------------------------------------------------
writeLines(capture.output(sessionInfo()), file.path("output", "session_info.txt"))

cat("=== Analysis complete ===\n")
cat("All outputs written to: output/\n")