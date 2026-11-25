# The proposal pipeline, in order: prepare_sessions -> compute.py (power + synchrony, one load per session) ->
# plot stages (proposal figures) -> proposal/img + proposal.pdf. Per-session caching stays inside compute.py.
#   zsh run_project.sh --dry-run        # what would run
#   zsh run_project.sh --smokescreen    # plot stages on 40 subjects of the existing cache -> *.smokescreen
#   zsh run_project.sh                  # full run
#   snakemake paper / simulations [--cores 4]   # paper figures (on hold) / validity checks -> figures/simulations/
#   add longetal=true for the Long et al. 2020 replication (power only, build_roi_power.py) -> *.longetal
import os, sys

configfile: "config/config.yaml"
workdir: workflow.basedir

SMOKE = str(config["smokescreen"]).lower() == "true"
LONG = str(config.get("longetal", False)).lower() == "true"
if SMOKE:
    os.environ["WOA_SMOKESCREEN"] = "1"   # fc.session_files: smokescreen_n_subjects random subjects
if LONG:
    os.environ["WOA_LONGETAL"] = "1"      # project_paths: Long et al. settings, scratch -> *.longetal
os.environ["CML_AUTO_APPROVE"] = "1"      # cml_data would otherwise prompt before downloading

from project_paths import SCRATCH_DIR
from simulate_eeg import SWEEPS

PY = sys.executable   # the env snakemake runs in
FIG = "figures" + (".longetal" if LONG else "") + (".smokescreen" if SMOKE else "")
DONE = f"{FIG.replace('figures', 'results', 1)}/.done"   # run markers: figures/ holds figures only
COMPUTED = [] if SMOKE else [f"{DONE}/real/compute"]   # smokescreen: the cache must exist
SWEEP_TAGS = [t for tags in SWEEPS.values() for t in tags]   # recovery sweeps: compute only
# paper (on hold) and simulations: plot script -> its figure subdirectory ("" = the script writes its own subdirectories)
PLOTS = {"plot_boxplots": "", "plot_spectrum": "burke_roi_power", "plot_epoch_network": "burke_roi_synchrony",
         "plot_latency": "latency", "plot_anticipation": "latency",
         "plot_decoding": "decoding", "plot_reliability": "reliability"}
# proposal/img name -> figure (proposal order)
PROPOSAL = {"figS1_power_subjects": "burke_roi_power/roi_power_word_on.png",                    # 1A
            "variance_components_power": "variance_structure/variance_components_power.png",  # 1B
            "fig1c_single_subject": "single_subject/single_subject_shares.png",              # 1C
            "variance_components_latency": "variance_structure/variance_components_latency.png",  # 2A
            "fig2b_propagation_waveforms": "timing/propagation_waveforms.png",               # 2B
            "fig2c_propagation_shift": "timing/propagation_shift.png",                       # 2C
            "fig4d_amp_latency": "timing/power_latency.png",                                 # 2D
            "fig4a_wait_waveforms": "timing/wait_waveform_occipital.png",                    # 3A
            "fig3b_offset_next": "timing/offset_next.png",                                   # 3B
            "fig3d_peaklat": "timing/isi_power_latency.png",                                 # 3C
            "carryover": "variance_structure/carryover.png",                                 # 3D
            "fig3e_isi_hist": "timing/isi_hist.png",                                         # 3E
            "fig4_recall_sme": "recall/recall_sme.png",                                      # 4
            "electrode_reliability": "variability/electrode_reliability.png",                # S1A
            "alpha_hg_across_electrodes": "variance_structure/alpha_hg_across_electrodes.png",  # S1B
            "figS2_offset_prev": "timing/offset_prev.png",                                   # S2
            "fig2c_subregion_waveforms": "timing/subregion_waveforms.png"}                   # S3


def sim(w): return "" if w.sim == "real" else f"--simulation-tag {w.sim}"   # sims -> <scratch>/sim/<tag>
def fig(w): return FIG if w.sim == "real" else f"{FIG}/simulations/{w.sim}"
def plot_cmds(w):
    return "; ".join(f"{PY} {s}.py {sim(w)} --out-dir {os.path.join(fig(w), d)}".rstrip("/") for s, d in PLOTS.items())


rule all:
    input: expand(f"{DONE}/real/{{band}}.long_power", band=[r["band"] for r in config["longetal_params"]["runs"]]) if LONG
           else f"{DONE}/proposal"

rule paper:
    input: f"{DONE}/real/plots"

rule simulations:
    input: expand(f"{DONE}/{{sim}}/plots", sim=config["simulations"]), f"{FIG}/simulations/recovery.png"

rule recovery:
    input: expand(f"{DONE}/{{sim}}/compute", sim=SWEEP_TAGS)
    output: f"{FIG}/simulations/recovery.png"
    shell: f"{PY} plot_recovery.py --out-dir {FIG}/simulations"

rule prepare_sessions:
    output: f"{SCRATCH_DIR}/sess_list_df.json"
    threads: workflow.cores
    shell: f"{PY} prepare_sessions.py --workers {{threads}}"

# threads = all cores, so compute runs one at a time (RAM, shared downloads).
rule compute:
    input: rules.prepare_sessions.output
    output: touch(f"{DONE}/{{sim}}/compute")
    threads: workflow.cores
    params: sim=sim
    shell: f"{PY} compute.py {{params.sim}} --workers {{threads}}"

# ── proposal figures ────────────────────────────────────────────────────────
rule power:          # 1A
    input: COMPUTED
    output: touch(f"{DONE}/real/power")
    shell: f"{PY} plot_boxplots.py --measure power --out-dir {FIG}"

rule variance:       # 1B, 2A, 3D, S1B; noise ceilings (R variance fits run once, --refit to redo)
    input: COMPUTED
    output: touch(f"{DONE}/real/variance")
    threads: workflow.cores
    shell: f"{PY} plot_variance_structure.py --cores {{threads}} --out-dir {FIG}/variance_structure"

rule single_subject: # 1C (needs the group fit)
    input: rules.variance.output
    output: touch(f"{DONE}/real/single_subject")
    threads: workflow.cores
    shell: f"{PY} plot_variance_structure.py --single --max-electrodes 100000 --cores {{threads}} --out-dir {FIG}/single_subject"

rule timing:         # 2B-D, 3A-C, 3E, S2, S3 (2D reads the variance tables)
    input: rules.variance.output
    output: touch(f"{DONE}/real/timing")
    shell: f"{PY} plot_timing.py --out-dir {FIG}/timing"

rule recall:         # 4 (R behavioural model once)
    input: COMPUTED
    output: touch(f"{DONE}/real/recall")
    threads: workflow.cores
    shell: f"{PY} plot_variance_structure.py --recall --cores {{threads}} --out-dir {FIG}/recall"

rule variability:    # S1A
    input: COMPUTED
    output: touch(f"{DONE}/real/variability")
    shell: f"{PY} plot_variability.py --out-dir {FIG}/variability"

rule proposal:       # copy the figures, recompile (full run only: the smokescreen leaves proposal/ alone)
    input: rules.power.output, rules.single_subject.output, rules.timing.output, rules.recall.output, rules.variability.output
    output: touch(f"{DONE}/proposal")
    run:
        if not SMOKE:
            for name, path in PROPOSAL.items():
                shell(f"cp {FIG}/{path} proposal/img/{name}.png")
            shell("cd proposal && tectonic --keep-logs --synctex proposal.tex")

# ── paper (on hold) and simulations ─────────────────────────────────────────
rule plots:
    input: f"{DONE}/{{sim}}/compute"
    output: touch(f"{DONE}/{{sim}}/plots")
    params: cmds=plot_cmds
    shell: "{params.cmds}"

rule long_power:   # Long et al. replication (closed): legacy compute + plot
    input: rules.prepare_sessions.output
    output: touch(f"{DONE}/real/{{band}}.long_power")
    threads: workflow.cores
    shell: f"{PY} build_roi_power.py --band {{wildcards.band}} --workers {{threads}} --out-dir {FIG}/burke_roi_power"
