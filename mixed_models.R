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
# variance and blank modes: the 13 random intercepts (DeHaan 2026 ch. 3's 10 + electrode, electrode in session, event)
add_terms <- function(tr) tr[, `:=`(elsess = paste(electrode, session), sess_list = paste(session, list),
                                    event = paste(session, event), subj_item = paste(subject, item),
                                    subj_sp = paste(subject, sp), list_sp = paste(list, sp), sess_sp = paste(session, sp))]
rand <- "(1 | item) + (1 | sp) + (1 | list) + (1 | subject) + (1 | subj_item) + (1 | session) + (1 | subj_sp) +
  (1 | list_sp) + (1 | sess_list) + (1 | sess_sp) + (1 | electrode) + (1 | elsess) + (1 | event)"

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
  tr <- add_terms(fread(file.path(path, "trials.csv")))
  f <- as.formula(paste("~", rand))
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
  # per lobe and band: y ~ prev_resp + prev2_resp + blank_len + prev_resp:blank_len + the variance model's 13 terms
  # prev_resp / prev2_resp: the response to the word 1 / 2 back at the same electrode (same list), z within
  # electrode-session (drift: both predict the word; carry-over: mostly the word 1 back); blank_len: blank before
  # the word, z within session. No electrode-in-list term: with ~11 words a list, a per-list offset biases a lagged
  # slope down by ~1/11 (Nickell 1981).
  suppressMessages(library(parallel))
  cores <- if (length(args) >= 3) as.integer(args[3]) else 6
  tr <- fread(file.path(path, "trials.csv"))[in_power == TRUE]
  setorder(tr, electrode, session, event)
  z <- function(v) (v - mean(v, na.rm = TRUE)) / sd(v, na.rm = TRUE)
  for (b in c("alpha", "high_gamma")) for (k in 1:2)
    tr[, paste0(b, "_prev", k) := z(fifelse(shift(event, k) == event - k & shift(list, k) == list, shift(get(b), k), NA_real_)),
       by = .(electrode, session)]
  lo <- c("750-1000" = 750, "800-1200" = 800); hi <- c("750-1000" = 1000, "800-1200" = 1200)
  tr <- tr[is.finite(blank_ms) & blank_ms >= lo[design] - 40 & blank_ms <= hi[design] + 40]   # screen-timing slop
  tr[, blank_len := z(blank_ms), by = session]
  add_terms(tr)
  jobs <- CJ(lobe = unique(tr$lobe), band = c("alpha", "high_gamma"))
  fits <- mclapply(seq_len(nrow(jobs)), function(i) {
    lb <- jobs$lobe[i]; b <- jobs$band[i]
    d <- tr[lobe == lb & is.finite(get(b))]
    d[, `:=`(prev_resp = get(paste0(b, "_prev1")), prev2_resp = get(paste0(b, "_prev2")))]
    d <- d[is.finite(prev_resp) & is.finite(prev2_resp)]
    if (uniqueN(d$subject) < 5) return(NULL)
    t0 <- Sys.time()
    m <- tryCatch(lmer(as.formula(paste(b, "~ prev_resp + prev2_resp + blank_len + prev_resp:blank_len +", rand)), data = d,
                       control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)),
                  error = function(e) { message(lb, " ", b, ": ", conditionMessage(e)); NULL })
    if (is.null(m)) return(NULL)
    message(lb, " ", b, ": ", nrow(d), " rows, ", round(difftime(Sys.time(), t0, units = "mins"), 1), " min")
    data.table(term = names(fixef(m)), est = fixef(m), se = sqrt(diag(as.matrix(vcov(m)))), lobe = lb, band = b,
               n = nrow(d), nsub = uniqueN(d$subject), nel = uniqueN(d$electrode))
  }, mc.cores = min(nrow(jobs), cores))
  fwrite(rbindlist(Filter(is.data.frame, fits)), file.path(path, "blank_fixed.csv"))
}
