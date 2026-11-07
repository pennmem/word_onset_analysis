# mixed_models.R -- mixed-effects tests on the per-electrode CSVs written by the Python
# plot scripts (word-on contrast).
#
#   Rscript mixed_models.R latency results/latency/hg_latency_word_on_electrodes.csv
#   Rscript mixed_models.R bands   results/burke_roi_power/roi_power_word_on_alpha_per_electrode.csv
#                                  (the high-gamma CSV is read from the same place)
#   Rscript mixed_models.R variance results/variance_structure [cores]
#   Rscript mixed_models.R blank    results/variance_structure [cores]   (both run by plot_variance_structure.py)
#
# Input: one row per electrode (sessions averaged) with its subject, Burke ROI ("L-frontal")
# and anatomical label ("L fusiform gyrus"). lobe = ROI without hemisphere; region = the
# anatomical label without hemisphere (hemispheres pooled, as the fine-region figures);
# electrodes without an anatomical label are dropped.
#   latency  latency_ms ~ lobe + (1|sub) + (1|sub:lobe) + (1|sub:region)
#            -> estimated latency per lobe; all 15 lobe pairs (Holm)
#   bands    cohens_d ~ lobe * band + (1 + band|sub) + (1|sub:lobe) + (1|sub:lobe:band)
#                       + (1|sub:region) + (1|sub:label)
#            band: alpha / high gamma (each electrode has one row per band, hence sub:label;
#            each subject its own alpha-gamma pattern, hence 1 + band|sub)
#            -> per lobe pair: does the alpha vs high-gamma pattern differ (lobe x band
#               interaction contrast, Holm over 15 pairs)
# sub:lobe (and sub:lobe:band): each subject has its own lobe effects, so lobe comparisons are
# tested against their variability across subjects, not across thousands of electrodes.
# Fine regions enter only within subject (sub:region: a subject's electrodes in one region are
# related). Not as (1|region): with 1-11 regions per lobe (hippocampus 1) the model cannot
# separate a lobe effect from its few regions and moves the lobe signal into them.
# Tests are asymptotic (z): with hundreds of subjects the df approximation does not matter.
# Output: <input stem>_mixed_*.csv next to the input.
suppressMessages({library(lme4); library(emmeans); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]; path <- args[2]
emm_options(lmer.df = "asymptotic")
load <- function(p) {
  d <- fread(p)
  d[, lobe := factor(sub(".*-", "", roi))]
  d[, region := sub("^[LR] ", "", fine)]
  d[!is.na(region) & !(region %in% c("", "nan", "None"))]
}

if (mode == "latency") {
  out <- sub("_electrodes\\.csv$", "_mixed", path)
  d <- load(path)
  m <- lmer(latency_ms ~ lobe + (1 | sub) + (1 | sub:lobe) + (1 | sub:region), data = d,
            control = lmerControl(optimizer = "bobyqa"))
  e <- emmeans(m, ~ lobe)
  pr <- pairs(e, adjust = "holm")
  fwrite(as.data.frame(e), paste0(out, "_lobes.csv"))
  fwrite(as.data.frame(pr), paste0(out, "_pairs.csv"))
  print(e); print(pr)
} else if (mode == "bands") {
  out <- sub("_alpha_per_electrode\\.csv$", "_mixed", path)
  d <- rbind(load(path)[, band := "alpha"],
             load(sub("_alpha_", "_high_gamma_", path))[, band := "high_gamma"])
  d[, band := factor(band)]
  m <- lmer(cohens_d ~ lobe * band + (1 + band | sub) + (1 | sub:lobe) + (1 | sub:lobe:band) +
              (1 | sub:region) + (1 | sub:label),
            data = d, control = lmerControl(optimizer = "bobyqa"))
  inter <- contrast(emmeans(m, ~ lobe * band), interaction = c("pairwise", "pairwise"), adjust = "holm")
  fwrite(as.data.frame(inter), paste0(out, "_pairs.csv"))
  print(joint_tests(m)); print(inter)
} else if (mode == "variance") {
  # path: plot_variance_structure.py's table directory. Per lobe and band (alpha, high gamma, high-gamma
  # latency), single word events, random intercepts only: DeHaan 2026 ch. 3's terms on the neural response
  # instead of recall,
  #   y ~ (1|item) + (1|sp) + (1|list) + (1|subject) + (1|subject:item) + (1|session) + (1|subject:sp)
  #       + (1|list:sp) + (1|session:list) + (1|session:sp)
  # plus the terms recall does not have: (1|electrode) + (1|electrode:session) + (1|event).
  # event: one word in one session, shared by the lobe's electrodes; the residual is what is
  # left per electrode and event. Share of a term = its variance / total. trials.csv holds the electrodes
  # the Python side chose (at most N per subject and lobe). Optional 3rd argument: parallel fits.
  suppressMessages(library(parallel))
  cores <- if (length(args) >= 3) as.integer(args[3]) else 6
  tr <- fread(file.path(path, "trials.csv"))
  tr[, `:=`(elsess = paste(electrode, session), sess_list = paste(session, list), event = paste(session, event),
            subj_item = paste(subject, item), subj_sp = paste(subject, sp), list_sp = paste(list, sp),
            sess_sp = paste(session, sp))]
  f <- ~ (1 | item) + (1 | sp) + (1 | list) + (1 | subject) + (1 | subj_item) + (1 | session) + (1 | subj_sp) +
    (1 | list_sp) + (1 | sess_list) + (1 | sess_sp) + (1 | electrode) + (1 | elsess) + (1 | event)
  jobs <- CJ(lobe = unique(tr$lobe), band = c("alpha", "high_gamma", "latency"))
  fits <- mclapply(seq_len(nrow(jobs)), function(i) {
    lb <- jobs$lobe[i]; b <- jobs$band[i]
    d <- tr[lobe == lb & is.finite(get(b)) & (b == "latency" | in_power)]   # latency: its own electrodes
    if (uniqueN(d$subject) < 5) return(NULL)
    t0 <- Sys.time()
    m <- tryCatch(lmer(update(f, as.formula(paste(b, "~ ."))), data = d,
                       control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)),
                  error = function(e) { message(lb, " ", b, ": ", conditionMessage(e)); NULL })
    if (is.null(m)) return(NULL)
    message(lb, " ", b, ": ", nrow(d), " rows, ", round(difftime(Sys.time(), t0, units = "mins"), 1), " min")
    as.data.table(VarCorr(m))[, .(term = grp, var = vcov)][, `:=`(lobe = lb, band = b, n = nrow(d),
                                                                   nsub = uniqueN(d$subject), nel = uniqueN(d$electrode))]
  }, mc.cores = min(nrow(jobs), cores))
  fwrite(rbindlist(Filter(is.data.frame, fits)), file.path(path, "components.csv"))
} else if (mode == "blank") {
  # per lobe and band: y ~ 0 + design + bin_c + prev_z + prev_z:blank_z + next_z + (1|subject) + (1|electrode)
  #   + (1|elsess) + (1|sp) + (1|sess_list)
  # bin_c: blank before the word, 50 ms bins, each vs its design's first bin; prev_z: the previous word's
  # response at the same electrode (same list, the immediately preceding word), z within electrode-session;
  # blank_z: blank before the word, z within session; next_z: the blank after the word (z within session;
  # a list's last word: 0), unknown when the word appears -> a control.
  suppressMessages(library(parallel))
  cores <- if (length(args) >= 3) as.integer(args[3]) else 6
  tr <- fread(file.path(path, "trials.csv"))
  tr[, `:=`(elsess = paste(electrode, session), sess_list = paste(session, list))]
  ev <- unique(tr[, .(session, list, event, blank_ms)])
  setorder(ev, session, event)
  ev[, next_blank := fifelse(shift(event, -1) == event + 1, shift(blank_ms, -1), NA_real_), by = .(session, list)]
  ev[, next_z := { v <- (next_blank - mean(next_blank, na.rm = TRUE)) / sd(next_blank, na.rm = TRUE); fifelse(is.na(v), 0, v) },
     by = session]
  tr <- merge(tr, ev[, .(session, event, next_z)], by = c("session", "event"))
  setorder(tr, electrode, session, event)
  lo <- c("750-1000" = 750, "800-1200" = 800); hi <- c("750-1000" = 1000, "800-1200" = 1200)
  tr <- tr[in_power & is.finite(blank_ms) & blank_ms >= lo[design] - 40 & blank_ms <= hi[design] + 40]   # screen-timing slop
  tr[, bin := lo[design] + 50 * pmin(floor((pmax(blank_ms, lo[design]) - lo[design]) / 50), (hi[design] - lo[design]) / 50 - 1)]
  tr[, bin_c := relevel(factor(fifelse(bin == lo[design], "ref", paste0(design, "_", bin))), "ref")]   # vs the design's first bin
  tr[, blank_z := (blank_ms - mean(blank_ms)) / sd(blank_ms), by = session]
  jobs <- CJ(lobe = unique(tr$lobe), band = c("alpha", "high_gamma"))
  fits <- mclapply(seq_len(nrow(jobs)), function(i) {
    lb <- jobs$lobe[i]; b <- jobs$band[i]
    d <- tr[lobe == lb & is.finite(get(b))]
    d[, prev := fifelse(shift(event) == event - 1 & shift(list) == list, shift(get(b)), NA_real_), by = elsess]
    d[, prev_z := (prev - mean(prev, na.rm = TRUE)) / sd(prev, na.rm = TRUE), by = elsess]
    d <- d[is.finite(prev_z)]
    if (uniqueN(d$subject) < 5) return(NULL)
    m <- tryCatch(lmer(as.formula(paste(b, "~ 0 + design + bin_c + prev_z + prev_z:blank_z + next_z + (1 | subject) + (1 | electrode) +",
                                        "(1 | elsess) + (1 | sp) + (1 | sess_list)")), data = d,
                       control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)),
                  error = function(e) { message(lb, " ", b, ": ", conditionMessage(e)); NULL })
    if (is.null(m)) return(NULL)
    data.table(term = names(fixef(m)), est = fixef(m), se = sqrt(diag(as.matrix(vcov(m)))), lobe = lb, band = b,
               n = nrow(d), nsub = uniqueN(d$subject), nel = uniqueN(d$electrode))
  }, mc.cores = min(nrow(jobs), cores))
  fwrite(rbindlist(Filter(is.data.frame, fits)), file.path(path, "blank_fixed.csv"))
}
