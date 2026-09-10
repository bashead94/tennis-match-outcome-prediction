# =============================================================================
# Exploratory Data Analysis — Assignment 2
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Package management
# -----------------------------------------------------------------------------
# pacman installs anything missing, so the script is self-contained in a sandbox.
# Only the packages actually used here are loaded.
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")

suppressPackageStartupMessages(
  pacman::p_load(
    readr,     # CSV reading
    dplyr,     # data manipulation
    ggplot2,   # plotting
    reshape2   # melt() for the correlation heatmap
  )
)


# -----------------------------------------------------------------------------
# 1. Paths & output folder  
# -----------------------------------------------------------------------------

DATA_DIR     <- "."
MATCHES_FILE <- "charting-m-matches.csv"
STATS_FILE   <- "charting-m-stats-overview.csv"

OUT_FIGS <- file.path("output", "figures")
dir.create(OUT_FIGS, recursive = TRUE, showWarnings = FALSE)

matches_path <- file.path(DATA_DIR, MATCHES_FILE)
stats_path   <- file.path(DATA_DIR, STATS_FILE)

if (!file.exists(matches_path) || !file.exists(stats_path)) {
  stop(
    paste0(
      "Data files not found. Place both files in the working directory (",
      normalizePath(DATA_DIR), "):\n",
      "  - ", MATCHES_FILE, "\n",
      "  - ", STATS_FILE, "\n",
      "File names are case-sensitive in a sandbox; rename to match exactly."
    ),
    call. = FALSE
  )
}

cat("=== Assignment 2: Exploratory Data Analysis ===\n\n")


# -----------------------------------------------------------------------------
# 2. Load data
# -----------------------------------------------------------------------------

tennis_data_1 <- suppressWarnings(read_csv(
  matches_path,
  col_types      = cols(.default = col_character()),
  show_col_types = FALSE
))

tennis_data_2 <- suppressWarnings(read_csv(
  stats_path,
  show_col_types = FALSE,
  name_repair    = "minimal"
))

cat("Matches rows:", nrow(tennis_data_1),
    "| Stats rows:", nrow(tennis_data_2), "\n\n")


# -----------------------------------------------------------------------------
# 3. Merge & clean
# -----------------------------------------------------------------------------

merged_data <- merge(tennis_data_1, tennis_data_2, by = "match_id")

cat("Merged rows:", nrow(merged_data), "| cols:", ncol(merged_data), "\n")

# Missing values per column (reproduces the report's Table 1, pre-cleaning)
cat("\nMissing values per column (merged, pre-clean):\n")
print(colSums(is.na(merged_data)))

merged_data_cleaned <- merged_data %>%
  filter(set == "Total") %>%
  filter(Surface %in% c("Clay", "Grass", "Hard", "Carpet")) %>%
  mutate(
    outcome         = ifelse(player == `Player 1`, "Win", "Loss"),
    year            = as.integer(substr(as.character(Date), 1, 4)),
    ace_rate        = ifelse(serve_pts > 0, aces       / serve_pts, NA_real_),
    df_rate         = ifelse(serve_pts > 0, dfs        / serve_pts, NA_real_),
    first_in_pct    = ifelse(serve_pts > 0, first_in   / serve_pts, NA_real_),
    first_won_pct   = ifelse(first_in  > 0, first_won  / first_in,  NA_real_),
    second_won_pct  = ifelse(second_in > 0, second_won / second_in, NA_real_),
    winners_fh_rate = ifelse(serve_pts > 0, winners_fh / serve_pts, NA_real_),
    winners_bh_rate = ifelse(serve_pts > 0, winners_bh / serve_pts, NA_real_),
    bp_save_rate    = ifelse(bk_pts    > 0, bp_saved   / bk_pts,    NA_real_)
  ) %>%
  filter(!is.na(first_won_pct), !is.na(second_won_pct))

cat("\nCleaned rows:", nrow(merged_data_cleaned), "\n")
cat("Outcome balance:\n"); print(table(merged_data_cleaned$outcome))
cat("\nSurface balance:\n"); print(table(merged_data_cleaned$Surface)); cat("\n")


# -----------------------------------------------------------------------------
# 4. Exploratory visuals  
# -----------------------------------------------------------------------------

show_and_save <- function(plot_obj, filename, width = 8, height = 5) {
  print(plot_obj)
  ggsave(file.path(OUT_FIGS, filename), plot = plot_obj,
         width = width, height = height, dpi = 150)
  cat("Saved:", file.path(OUT_FIGS, filename), "\n")
}

# Fig 1 — Match distribution by surface (categorical)
p1 <- ggplot(merged_data_cleaned, aes(x = Surface)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Match Distribution by Surface", x = "Surface", y = "Count")
show_and_save(p1, "01_match_distribution_by_surface.png")

# Fig 2 — Match coverage by year
p2 <- ggplot(merged_data_cleaned, aes(x = year)) +
  geom_histogram(bins = 30, fill = "steelblue", colour = "white") +
  labs(title = "Match Coverage by Year", x = "Year", y = "Count")
show_and_save(p2, "02_match_coverage_by_year.png")

# Fig 3 — Distribution of ace rate (continuous)
p3 <- ggplot(filter(merged_data_cleaned, !is.na(ace_rate)), aes(x = ace_rate)) +
  geom_histogram(bins = 40, fill = "darkorange", colour = "white") +
  labs(title = "Distribution of Ace Rate", x = "Aces per Serve Point", y = "Frequency")
show_and_save(p3, "03_distribution_of_ace_rate.png")

# Fig 4 — Ace rate by surface
p4 <- ggplot(filter(merged_data_cleaned, !is.na(ace_rate)),
             aes(x = Surface, y = ace_rate, fill = Surface)) +
  geom_boxplot() +
  labs(title = "Ace Rate by Surface", x = "Surface", y = "Ace Rate")
show_and_save(p4, "04_ace_rate_by_surface.png")

# Fig 5 — Ace rate by match outcome
p5 <- ggplot(filter(merged_data_cleaned, !is.na(ace_rate)),
             aes(x = outcome, y = ace_rate, fill = outcome)) +
  geom_boxplot() +
  labs(title = "Ace Rate by Match Outcome", x = "Outcome", y = "Ace Rate")
show_and_save(p5, "05_ace_rate_by_outcome.png")

# Fig 6 — First serve win % by outcome
p6 <- ggplot(merged_data_cleaned,
             aes(x = outcome, y = first_won_pct, fill = outcome)) +
  geom_boxplot() +
  labs(title = "First Serve Win % by Outcome", x = "Outcome", y = "First Serve Win %")
show_and_save(p6, "06_first_serve_win_pct_by_outcome.png")

# Fig 7 — Break point save rate by outcome

p7 <- ggplot(filter(merged_data_cleaned, !is.na(bp_save_rate)),
             aes(x = outcome, y = bp_save_rate, fill = outcome)) +
  geom_boxplot() +
  labs(title = "Break Point Save Rate by Outcome", x = "Outcome", y = "BP Save Rate")
show_and_save(p7, "07_bp_save_rate_by_outcome.png")

# Fig 8 — Correlation matrix of engineered features
numeric_features <- merged_data_cleaned %>%
  select(ace_rate, df_rate, first_in_pct, first_won_pct,
         second_won_pct, winners_fh_rate, winners_bh_rate, bp_save_rate) %>%
  na.omit()

cor_matrix <- cor(numeric_features)
melted_cor <- melt(cor_matrix)

p8 <- ggplot(melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Matrix of Engineered Features", x = "Var1", y = "Var2")
show_and_save(p8, "08_correlation_matrix.png")


# -----------------------------------------------------------------------------
# 5. Done
# -----------------------------------------------------------------------------
cat("\n=== EDA complete ===\n")
cat("Figures saved to:", OUT_FIGS, "\n")
