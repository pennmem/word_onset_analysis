# Single-subject variance fits (Riley): the Fig. 1 model within one subject, subject-level terms dropped; all of the
# subject's occipital electrodes (trials.csv rebuilt without the 5-electrode cap); subjects with >= 5 electrodes and
# >= 2 sessions; region only where the subject has >= 2 regions. Alpha and high gamma. -> components_single.csv
#   Rscript single_subject_variance.R results/single_subject   (trials.csv there from plot_variance_structure.py
#   --max-electrodes 100000 --out-dir figures/single_subject, stopped before its model fits)
suppressMessages({library(lme4); library(data.table); library(parallel)})
dir <- commandArgs(trailingOnly = TRUE)[1]
tr <- fread(file.path(dir, "trials.csv"), select = c("electrode", "subject", "session", "list", "sp", "item", "event",
                                                    "lobe", "region", "alpha", "high_gamma"))[lobe == "occipital"]
tr[, `:=`(elsess = paste(electrode, session), sess_list = paste(session, list), event = paste(session, event),
          list_sp = paste(list, sp), sess_sp = paste(session, sp))]
ok <- tr[, .(nel = uniqueN(electrode), nses = uniqueN(session)), by = subject][nel >= 5 & nses >= 2, subject]
jobs <- CJ(subject = ok, band = c("alpha", "high_gamma"))
message(length(ok), " subjects, ", nrow(jobs), " fits")
fits <- mclapply(seq_len(nrow(jobs)), function(i) {
  d <- tr[subject == jobs$subject[i] & is.finite(get(jobs$band[i]))]
  reg <- uniqueN(na.omit(d$region)) >= 2
  f <- paste(jobs$band[i], "~ (1 | item) + (1 | sp) + (1 | list) + (1 | session) + (1 | list_sp) + (1 | sess_list) +",
             "(1 | sess_sp) + (1 | electrode) + (1 | elsess) + (1 | event)", if (reg) "+ (1 | region)" else "")
  m <- tryCatch(lmer(as.formula(f), data = if (reg) d[!is.na(region)] else d,
                     control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  as.data.table(VarCorr(m))[, .(term = grp, var = vcov)][, `:=`(subject = jobs$subject[i], band = jobs$band[i],
                                                                 nel = uniqueN(d$electrode), nses = uniqueN(d$session), n = nrow(d))]
}, mc.cores = 8)
fwrite(rbindlist(Filter(is.data.frame, fits)), file.path(dir, "components_single.csv"))
