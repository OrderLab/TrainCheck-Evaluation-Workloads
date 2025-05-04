xonsh run_all.xsh -v --res_folder perf_res_ae
xonsh analysis.xsh --res_folder perf_res_ae
python3 plot_e2e.py -o performance_ae.pdf -i perf_res_ae/overhead_e2e.csv -t OSDIAE