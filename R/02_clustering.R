# =============================================================================
# Tennis Match Profile Analysis: PCA + K-Means Clustering
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Package Management
# -----------------------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")

pacman::p_load(
  readr,        # Fast CSV reading
  ggplot2,      # Grammar of graphics plotting
  factoextra,   # PCA and cluster visualisation helpers
  ggrepel,      # Non-overlapping label placement
  here,         # Relative file paths anchored to project root
  dplyr,        # Data manipulation
  cluster,      # Silhouette analysis
  tidyr         # Data tidying
)


# -----------------------------------------------------------------------------
# 1. Project Path Configuration
# -----------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
DATA_PATH    <- "charting-m-stats-overview.csv"
OUT_FIGS     <- "output/figures"
OUT_TABLES   <- "output/tables"

# Create output directories
dir.create(OUT_FIGS,   recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_TABLES, recursive = TRUE, showWarnings = FALSE)

# Confirm paths to the console
cat("=== Path Configuration ===\n")
cat("Data  :", DATA_PATH, "\n")
cat("Figs  :", OUT_FIGS,  "\n")
cat("Tables:", OUT_TABLES,"\n\n")


# -----------------------------------------------------------------------------
# 2. Load & Clean Data
# -----------------------------------------------------------------------------
tennis_data <- read_csv(DATA_PATH, show_col_types = FALSE)

# Keep only "Total" rows to avoid double-counting individual sets
tennis_total <- subset(tennis_data, set == "Total")

# Convert raw counts to rates and percentages
tennis_total <- tennis_total %>%
  mutate(
    total_points          = serve_pts + return_pts,
    ace_rate              = aces          / serve_pts,
    double_fault_rate     = dfs           / serve_pts,
    first_serve_pct       = first_in      / serve_pts,
    winner_rate           = winners       / total_points,
    unforced_error_rate   = unforced      / total_points,
    return_pts_won_pct    = return_pts_won / return_pts,
    bp_save_rate          = bp_saved      / bk_pts
  )

# Diagnostic: print mean and max serve points to console
mean_serve <- mean(tennis_total$serve_pts, na.rm = TRUE)
max_serve  <- max(tennis_total$serve_pts,  na.rm = TRUE)
cat(sprintf("Mean serve_pts: %.1f  |  Max serve_pts: %.0f\n\n", mean_serve, max_serve))

# Remove extreme outliers (most notably the 11-hour from 2010 match)
outliers <- subset(tennis_total, serve_pts > 465)
cat("Removing the following extreme outlier matches:\n")
print(outliers[, c("match_id", "player", "serve_pts")])
tennis_total <- subset(tennis_total, serve_pts <= 465)

# Preserve text metadata for later player mapping
metadata <- tennis_total[, c("match_id", "player", "set")]

# Extract the 7 numerical feature columns
numeric_features <- tennis_total[, c(
  "ace_rate", "double_fault_rate", "first_serve_pct",
  "winner_rate", "unforced_error_rate", "return_pts_won_pct", "bp_save_rate"
)]

# Remove rows with any NAs (required for PCA and K-Means)
complete_rows    <- complete.cases(numeric_features)
numeric_features <- numeric_features[complete_rows, ]
metadata         <- metadata[complete_rows, ]

cat(sprintf("\nRows after cleaning: %d\n\n", nrow(numeric_features)))


# -----------------------------------------------------------------------------
# 3. PCA
# -----------------------------------------------------------------------------
# center = TRUE and scale. = TRUE standardise features before decomposition
pca_model <- prcomp(numeric_features, center = TRUE, scale. = TRUE)

cat("=== PCA Variance Summary ===\n")
print(summary(pca_model))

cat("\n=== Loadings (PC1–PC4) ===\n")
print(pca_model$rotation[, 1:4])


# -- Scree Plot ---------------------------------------------------------------
png(file.path(OUT_FIGS, "01_scree_plot.png"), width = 800, height = 500)
plot(pca_model, type = "l", main = "Scree Plot: Explained Variance by Component")
dev.off()
cat("Saved: 01_scree_plot.png\n")


# -- Standard Biplot ----------------------------------------------------------
png(file.path(OUT_FIGS, "02_pca_biplot_standard.png"), width = 900, height = 700)
biplot(pca_model, scale = 0, main = "Tennis PCA Biplot")
dev.off()
cat("Saved: 02_pca_biplot_standard.png\n")


# -- Loadings Vector Plot -----------------------------------------------------
loadings <- pca_model$rotation

png(file.path(OUT_FIGS, "03_pca_loadings_plot.png"), width = 700, height = 700)
plot(loadings[, 1:2], type = "n",
     xlim = c(-1.1, 1.1), ylim = c(-1.1, 1.1),
     xlab = "PC1", ylab = "PC2",
     main = "Tennis PCA Loadings")
symbols(0, 0, circles = 1, inches = FALSE, add = TRUE, fg = "grey")
arrows(0, 0, loadings[, 1], loadings[, 2], col = "blue", length = 0.1)
text(loadings[, 1], loadings[, 2], labels = rownames(loadings), pos = 3, cex = 0.8)
dev.off()
cat("Saved: 03_pca_loadings_plot.png\n")


# Extract PC1 and PC2 scores for downstream clustering
pca_2d_data <- pca_model$x[, 1:2]


# -----------------------------------------------------------------------------
# 4. K-Means Validation: Elbow (WSS) + Silhouette
# -----------------------------------------------------------------------------

# -- WSS Elbow Method ---------------------------------------------------------
wss <- numeric(10)
for (i in 1:10) {
  set.seed(42)
  km_test <- kmeans(pca_2d_data, centers = i, nstart = 25,
                    iter.max = 300, algorithm = "MacQueen")
  wss[i] <- km_test$tot.withinss
}

png(file.path(OUT_FIGS, "04_elbow_wss.png"), width = 800, height = 500)
plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of Clusters K",
     ylab = "Total Within-Cluster Sum of Squares (WSS)",
     main = "K-Means Elbow Method: Finding Optimal Clusters")
dev.off()
cat("Saved: 04_elbow_wss.png\n")

# Export WSS values as a table
wss_table <- data.frame(K = 1:10, WSS = wss)
write.csv(wss_table, file.path(OUT_TABLES, "wss_by_k.csv"), row.names = FALSE)
cat("Saved: wss_by_k.csv\n")


# -- Silhouette Scores (K = 2 to 10) -----------------------------------------
sil_scores <- numeric(9)

for (i in 2:10) {
  set.seed(42)
  km <- kmeans(pca_2d_data, centers = i, nstart = 25,
               iter.max = 300, algorithm = "MacQueen")
  sil_scores[i - 1] <- mean(silhouette(km$cluster, dist(pca_2d_data))[, 3])
}

png(file.path(OUT_FIGS, "05_silhouette_scores.png"), width = 800, height = 500)
plot(2:10, sil_scores, type = "b", pch = 19,
     xlab = "Number of Clusters K",
     ylab = "Average Silhouette Width",
     main = "Silhouette Analysis for Optimal K")
dev.off()
cat("Saved: 05_silhouette_scores.png\n")

# Export silhouette scores as a table
sil_table <- data.frame(K = 2:10, Avg_Silhouette_Width = sil_scores)
write.csv(sil_table, file.path(OUT_TABLES, "silhouette_scores_by_k.csv"), row.names = FALSE)
cat("Saved: silhouette_scores_by_k.csv\n")

cat("\n=== Silhouette Scores by K ===\n")
print(sil_table)


# -- Silhouette Plot for Final K = 3 ------------------------------------------
set.seed(42)
kmeans_3 <- kmeans(pca_2d_data, centers = 3, nstart = 25,
                   iter.max = 300, algorithm = "MacQueen")
sil_3 <- silhouette(kmeans_3$cluster, dist(pca_2d_data))

cat("\n=== Silhouette Summary (K = 3) ===\n")
print(summary(sil_3))

png(file.path(OUT_FIGS, "06_silhouette_k3.png"), width = 800, height = 500)
plot(sil_3,
     main   = "Silhouette Plot – 3 Clusters",
     border = NA,
     col    = c("skyblue2", "tomato2", "palegreen3"))
dev.off()
cat("Saved: 06_silhouette_k3.png\n")


# -- Export silhouette summary stats ------------------------------------------
sil_summary <- summary(sil_3)
sil_summary_table <- data.frame(
  Cluster             = 1:3,
  Cluster_Size        = as.vector(sil_summary$clus.sizes),
  Avg_Silhouette_Width = as.vector(sil_summary$clus.avg.widths)
)
sil_summary_table$Overall_Avg <- sil_summary$avg.width  # repeated for reference

write.csv(sil_summary_table,
          file.path(OUT_TABLES, "silhouette_summary_k3.csv"),
          row.names = FALSE)
cat("Saved: silhouette_summary_k3.csv\n")


# -----------------------------------------------------------------------------
# 5. Cluster Profiling (PCA-space clusters)
# -----------------------------------------------------------------------------
tennis_profiled <- numeric_features %>%
  mutate(Cluster = as.factor(kmeans_3$cluster))

cluster_profile_table <- tennis_profiled %>%
  group_by(Cluster) %>%
  summarise(
    Match_Count             = n(),
    Avg_Ace_Rate            = mean(ace_rate,            na.rm = TRUE),
    Avg_DF_Rate             = mean(double_fault_rate,   na.rm = TRUE),
    Avg_1st_Serve_Pct       = mean(first_serve_pct,     na.rm = TRUE),
    Avg_Winner_Rate         = mean(winner_rate,         na.rm = TRUE),
    Avg_Unforced_Error_Rate = mean(unforced_error_rate, na.rm = TRUE),
    Avg_Return_Pts_Won_Pct  = mean(return_pts_won_pct,  na.rm = TRUE),
    Avg_BP_Save_Rate        = mean(bp_save_rate,        na.rm = TRUE)
  )

cat("\n=== Cluster Profile Table (K = 3, PCA-space) ===\n")
print(cluster_profile_table)

write.csv(cluster_profile_table,
          file.path(OUT_TABLES, "cluster_profile_k3.csv"),
          row.names = FALSE)
cat("Saved: cluster_profile_k3.csv\n")


# -----------------------------------------------------------------------------
# 6. Final K-Means on Scaled Full-Feature Space
# -----------------------------------------------------------------------------
scaled_features <- scale(numeric_features)

set.seed(42)
kmeans_model <- kmeans(scaled_features, centers = 3, nstart = 25)

metadata$Cluster <- as.factor(kmeans_model$cluster)

profile_clusters <- cbind(numeric_features, Cluster = kmeans_model$cluster)
cluster_means    <- aggregate(. ~ Cluster, data = profile_clusters, FUN = mean)

cat("\n=== Transposed 3-Cluster Profile Means (Scaled Full Space) ===\n")
print(t(cluster_means))

write.csv(cluster_means,
          file.path(OUT_TABLES, "cluster_means_scaled.csv"),
          row.names = FALSE)
cat("Saved: cluster_means_scaled.csv\n")


# -----------------------------------------------------------------------------
# 7. Cluster Visualisations
# -----------------------------------------------------------------------------

# -- K-Means cluster plot in PCA space ----------------------------------------
p_cluster <- fviz_cluster(kmeans_model, data = scaled_features,
                          geom         = "point",
                          ellipse.type = "convex",
                          ggtheme      = theme_minimal(),
                          main         = "K-Means Clustering of Tennis Match Profiles (3 Clusters)")

ggsave(file.path(OUT_FIGS, "07_kmeans_cluster_plot.png"),
       plot = p_cluster, width = 10, height = 7, dpi = 150)
cat("Saved: 07_kmeans_cluster_plot.png\n")


# -- Variable Vector Biplot over Clusters -------------------------------------
p_biplot <- fviz_pca_biplot(pca_model,
                            axes         = c(1, 2),
                            col.ind      = metadata$Cluster,
                            palette      = "Set1",
                            label        = "var",
                            col.var      = "black",
                            repel        = TRUE,
                            legend.title = "Clusters",
                            title        = "Tennis Biplot: Variable Vector Map over 3 Clusters")

ggsave(file.path(OUT_FIGS, "08_pca_biplot_clusters.png"),
       plot = p_biplot, width = 10, height = 7, dpi = 150)
cat("Saved: 08_pca_biplot_clusters.png\n")


# -----------------------------------------------------------------------------
# 8. Player Profile Footprint Plots
# -----------------------------------------------------------------------------
plot_data <- data.frame(
  match_id = metadata$match_id,
  player   = metadata$player,
  PC1      = pca_model$x[, 1],
  PC2      = pca_model$x[, 2],
  Cluster  = metadata$Cluster
)

target_players <- c("Roger Federer", "Rafael Nadal", "Marin Cilic")

for (current_player in target_players) {
  
  player_matches <- subset(plot_data, player == current_player)
  player_average <- aggregate(cbind(PC1, PC2) ~ player, data = player_matches, FUN = mean)
  
  cat(paste0("\n--- Career Average Coordinates for ", current_player, " ---\n"))
  print(player_average)
  
  p <- ggplot() +
    geom_point(data = plot_data,
               aes(x = PC1, y = PC2),
               color = "gray90", alpha = 0.2, size = 1) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray60", alpha = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray60", alpha = 0.5) +
    geom_point(data = player_matches,
               aes(x = PC1, y = PC2, color = player),
               alpha = 0.4, size = 2) +
    geom_point(data = player_average,
               aes(x = PC1, y = PC2, fill = player),
               color = "black", shape = 24, size = 5, stroke = 1.5) +
    geom_label_repel(data = player_average,
                     aes(x = PC1, y = PC2,
                         label = paste(player, "Career Avg")),
                     color = "black", fontface = "bold", fill = "white",
                     box.padding = 0.6, segment.color = "black",
                     show.legend = FALSE) +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1") +
    labs(
      title    = paste("Player Profile Footprint:", current_player),
      subtitle = "Individual matches and career average",
      x        = "PC1: Higher values broadly associated with attacking play and fewer errors",
      y        = "PC2: Higher values broadly associated with serving risk-taking",
      color    = "Individual Matches",
      fill     = "Career Average"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title   = element_text(face = "bold", size = 14),
      axis.title   = element_text(face = "bold"),
      legend.position = "bottom"
    ) +
    coord_cartesian(xlim = c(-5, 15), ylim = c(-5, 5))
  
  print(p)
  
  # Build a safe filename from the player name (replace spaces with underscores)
  safe_name <- gsub(" ", "_", current_player)
  fig_path  <- file.path(OUT_FIGS, paste0("09_player_footprint_", safe_name, ".png"))
  ggsave(fig_path, plot = p, width = 10, height = 7, dpi = 150)
  cat(paste0("Saved: 09_player_footprint_", safe_name, ".png\n"))
}


# -----------------------------------------------------------------------------
# 9. Session Information
# -----------------------------------------------------------------------------
# Captures R version, OS, and exact package versions for full reproducibility
session_path <- here("output", "session_info.txt")
sink(session_path)
cat("=== Session Information ===\n")
cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
print(sessionInfo())
sink()
cat(paste0("\nSession info saved to: ", session_path, "\n"))

cat("\n=== Analysis Complete ===\n")
cat("All outputs written to: output/\n")