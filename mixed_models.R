# mixed_models.R -- mixed-effects models on CSVs written by the Python plot scripts (word-on contrast).
#
#   Rscript mixed_models.R latency results/latency/hg_latency_word_on_electrodes.csv
#   Rscript mixed_models.R bands   results/burke_roi_power/roi_power_word_on_alpha_per_electrode.csv
#                                  (the high-gamma CSV is read from the same place)
#   Rscript mixed_models.R variance results/variance_structure [cores]   (plot_variance_structure.py)
#   Rscript mixed_models.R single   results/single_subject [cores]       (plot_variance_structure.py --single)
#   Rscript mixed_models.R recall   results/recall [cores]               (plot_variance_structure.py --recall)
#
# latency, bands: one row per electrode (sessions averaged); lobe = ROI without hemisphere, region = anatomical
# label without hemisphere; electrodes without a label are dropped.
#   latency  latency_ms ~ lobe + (1|sub) + (1|sub:lobe) + (1|sub:region) -> latency per lobe, 15 lobe pairs (Holm)
#   bands    cohens_d ~ lobe * band + (1 + band|sub) + (1|sub:lobe) + (1|sub:lobe:band) + (1|sub:region) + (1|sub:label)
#            -> per lobe pair, lobe x band interaction contrast (Holm)
# sub:lobe tests lobe differences against their variability across subjects, not across electrodes. Regions enter
# only within subject: as (1|region), with 1-11 regions per lobe, the lobe effect would move into its regions.
# Tests are asymptotic (z). Output: <input stem>_mixed_*.csv next to the input.
suppressMessages({library(lme4); library(emmeans); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]; path <- args[2]
emm_options(lmer.df = "asymptotic")
load <- function(csv) {
  electrodes <- fread(csv)
  electrodes[, lobe := factor(sub(".*-", "", roi))]
  electrodes[, region := sub("^[LR] ", "", fine)]
  electrodes[!is.na(region) & !(region %in% c("", "nan", "None"))]
}
# DeHaan 2026 ch. 3's 10 task terms + electrode, electrode:session and word event
add_terms <- function(trials) {
  if ("electrode" %in% names(trials)) trials[, elsess := paste(electrode, session)]
  trials[, `:=`(sess_list = paste(session, list), event = paste(session, event), subj_item = paste(subject, item),
                subj_sp = paste(subject, sp), list_sp = paste(list, sp), sess_sp = paste(session, sp))]
}
rand <- "(1 | item) + (1 | sp) + (1 | list) + (1 | subject) + (1 | subj_item) + (1 | session) + (1 | subj_sp) +
  (1 | list_sp) + (1 | sess_list) + (1 | sess_sp) + (1 | electrode) + (1 | elsess) + (1 | event)"
cores <- if (length(args) >= 3) as.integer(args[3]) else 6
fit_minutes <- function(start) round(difftime(Sys.time(), start, units = "mins"), 1)

if (mode == "latency") {
  out <- sub("_electrodes\\.csv$", "_mixed", path)
  electrodes <- load(path)
  model <- lmer(latency_ms ~ lobe + (1 | sub) + (1 | sub:lobe) + (1 | sub:region), data = electrodes,
                control = lmerControl(optimizer = "bobyqa"))
  lobe_means <- emmeans(model, ~ lobe)
  lobe_pairs <- pairs(lobe_means, adjust = "holm")
  fwrite(as.data.frame(lobe_means), paste0(out, "_lobes.csv"))
  fwrite(as.data.frame(lobe_pairs), paste0(out, "_pairs.csv"))
  print(lobe_means); print(lobe_pairs)
} else if (mode == "bands") {
  out <- sub("_alpha_per_electrode\\.csv$", "_mixed", path)
  electrodes <- rbind(load(path)[, band := "alpha"],
                      load(sub("_alpha_", "_high_gamma_", path))[, band := "high_gamma"])
  electrodes[, band := factor(band)]
  model <- lmer(cohens_d ~ lobe * band + (1 + band | sub) + (1 | sub:lobe) + (1 | sub:lobe:band) +
                  (1 | sub:region) + (1 | sub:label),
                data = electrodes, control = lmerControl(optimizer = "bobyqa"))
  interaction_pairs <- contrast(emmeans(model, ~ lobe * band), interaction = c("pairwise", "pairwise"), adjust = "holm")
  fwrite(as.data.frame(interaction_pairs), paste0(out, "_pairs.csv"))
  print(joint_tests(model)); print(interaction_pairs)
} else if (mode == "variance") {
  # per lobe and measure, single word events: y ~ rand (+ (1|region) where the lobe has >= 2 subregions).
  # Share of a term = its variance / total. trials.csv: the electrodes plot_variance_structure.py chose.
  suppressMessages(library(parallel))
  trials <- add_terms(fread(file.path(path, "trials.csv")))
  jobs <- CJ(lobe = unique(trials$lobe), band = c("alpha", "high_gamma", "latency"))
  fits <- mclapply(seq_len(nrow(jobs)), function(i) {
    lobe_name <- jobs$lobe[i]; band <- jobs$band[i]
    rows <- trials[lobe == lobe_name & is.finite(get(band)) & (band == "latency" | in_power)]   # latency: its own electrodes
    if (uniqueN(rows$subject) < 5) return(NULL)
    start <- Sys.time()
    region_term <- if (uniqueN(rows$region) >= 2) "+ (1 | region)" else ""
    model <- tryCatch(lmer(as.formula(paste(band, "~", rand, region_term)), data = rows,
                           control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)),
                      error = function(e) { message(lobe_name, " ", band, ": ", conditionMessage(e)); NULL })
    if (is.null(model)) return(NULL)
    message(lobe_name, " ", band, ": ", nrow(rows), " rows, ", fit_minutes(start), " min")
    as.data.table(VarCorr(model))[, .(term = grp, var = vcov)][, `:=`(lobe = lobe_name, band = band, n = nrow(rows),
                                                                       nsub = uniqueN(rows$subject), nel = uniqueN(rows$electrode))]
  }, mc.cores = min(nrow(jobs), cores))
  fwrite(rbindlist(Filter(is.data.frame, fits)), file.path(path, "components.csv"))
} else if (mode == "single") {
  # the variance model within one subject (Riley), occipital, all the subject's electrodes; subjects with >= 5
  # electrodes and >= 2 sessions; (1|region) where the subject has >= 2 subregions
  suppressMessages(library(parallel))
  trials <- add_terms(fread(file.path(path, "trials.csv"), select = c("electrode", "subject", "session", "list", "sp", "item",
                                                                     "event", "lobe", "region", "alpha", "high_gamma"))[lobe == "occipital"])
  within_subject_terms <- gsub("\\(1 \\| (subject|subj_item|subj_sp)\\) \\+\\s*", "", rand)
  eligible <- trials[, .(nel = uniqueN(electrode), nses = uniqueN(session)), by = subject][nel >= 5 & nses >= 2, subject]
  jobs <- CJ(subject = eligible, band = c("alpha", "high_gamma"))
  message(length(eligible), " subjects, ", nrow(jobs), " fits")
  fits <- mclapply(seq_len(nrow(jobs)), function(i) {
    rows <- trials[subject == jobs$subject[i] & is.finite(get(jobs$band[i]))]
    has_regions <- uniqueN(na.omit(rows$region)) >= 2
    model <- tryCatch(lmer(as.formula(paste(jobs$band[i], "~", within_subject_terms, if (has_regions) "+ (1 | region)" else "")),
                           data = if (has_regions) rows[!is.na(region)] else rows,
                           control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)), error = function(e) NULL)
    if (is.null(model)) return(NULL)
    as.data.table(VarCorr(model))[, .(term = grp, var = vcov)][, `:=`(subject = jobs$subject[i], band = jobs$band[i],
                                                                       nel = uniqueN(rows$electrode), nses = uniqueN(rows$session),
                                                                       n = nrow(rows))]
  }, mc.cores = cores)
  fwrite(rbindlist(Filter(is.data.frame, fits)), file.path(path, "components_single.csv"))
} else if (mode == "recall") {
  # DeHaan 2026 ch. 3.4.3 / 3.4.8: per experiment, recalled ~ his 10 task terms (logistic); predicted recall per event
  # from the terms he kept for the offsets (Item, SP, List, Subject, List:SP, Subject:Session, Subject:SP)
  suppressMessages(library(parallel))
  words <- fread(file.path(path, "behaviour.csv"))[is.finite(recalled)][, event_index := event]
  words <- add_terms(words)
  task_terms <- gsub("\\s*\\+ \\(1 \\| (electrode|elsess|event)\\)", "", rand)
  offset_terms <- c("item", "sp", "list", "subject", "list_sp", "session", "subj_sp")
  fits <- mclapply(unique(words$experiment), function(experiment_name) {
    rows <- words[experiment == experiment_name]
    start <- Sys.time()
    model <- tryCatch(glmer(as.formula(paste("recalled ~", task_terms)), data = rows, family = binomial,
                            control = glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)),
                      error = function(e) { message(experiment_name, ": ", conditionMessage(e)); NULL })
    if (is.null(model)) return(NULL)
    message(experiment_name, ": ", nrow(rows), " events, ", fit_minutes(start), " min")
    intercepts <- ranef(model)
    logit <- fixef(model)[1] + Reduce(`+`, lapply(offset_terms, function(term) intercepts[[term]][as.character(rows[[term]]), 1]))
    rows[, .(session, event = event_index, phat = plogis(logit))]
  }, mc.cores = cores)
  fwrite(rbindlist(Filter(is.data.frame, fits)), file.path(path, "recall_phat.csv"))
}
