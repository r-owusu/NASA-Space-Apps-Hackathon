#!/usr/bin/env python3
import os, json, time, argparse
from itertools import combinations
from math import ceil
import numpy as np, pandas as pd
from tqdm import tqdm
from utils.physics_utils import angular_separation
from utils.data_utils import save_csv, plot_histograms

MESSENGERS = ["GW","Gamma","Neutrino","Optical","Radio"]
MESS_PARAMS = {
    "GW": {"pos_err":10.0,"strength_mu":20,"strength_sigma":5,"time_jitter":30,"detect_prob":0.5},
    "Gamma": {"pos_err":5.0,"strength_mu":15,"strength_sigma":4,"time_jitter":20,"detect_prob":0.6},
    "Neutrino": {"pos_err":2.0,"strength_mu":8,"strength_sigma":2,"time_jitter":15,"detect_prob":0.4},
    "Optical": {"pos_err":0.3,"strength_mu":18,"strength_sigma":3,"time_jitter":120,"detect_prob":0.7},
    "Radio": {"pos_err":0.5,"strength_mu":12,"strength_sigma":3,"time_jitter":60,"detect_prob":0.5},
}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def simulate_true_sources(n_true_sources, time_span_seconds, rng):
    src_times = rng.uniform(0, time_span_seconds, size=n_true_sources)
    src_ras = rng.uniform(0, 360, size=n_true_sources)
    u = rng.uniform(-1,1,size=n_true_sources)
    src_decs = np.rad2deg(np.arcsin(u))
    src_ids = [f"SRC_{i:07d}" for i in range(n_true_sources)]
    return pd.DataFrame({"source_id":src_ids,"base_time":src_times,"base_ra":src_ras,"base_dec":src_decs})

def simulate_messenger(m, n_events, df_sources, time_span_seconds, rng, params):
    detections = []
    src_mask = rng.rand(len(df_sources)) < params.get("detect_prob",0.5)
    detected = df_sources[src_mask].reset_index(drop=True)
    for i,row in detected.iterrows():
        srcid=row['source_id']; bt=row['base_time']; bra=row['base_ra']; bdec=row['base_dec']
        t = rng.normal(bt, params['time_jitter']); ra = rng.normal(bra, params['pos_err']/3.0)
        dec = rng.normal(bdec, params['pos_err']/3.0)
        pos_err = abs(rng.normal(params['pos_err'], params['pos_err']*0.1))
        strength = abs(rng.normal(params['strength_mu'], params['strength_sigma']))
        detections.append({"event_id":f"{m}_true_{srcid}","messenger":m,"time":float(max(0,t)),"ra":float(ra%360),
                           "dec":float(max(-90,min(90,dec))),"pos_err":float(pos_err),"strength":float(strength),
                           "duration":float(rng.exponential(5.0)),"label":1,"source_id":srcid})
    n_true=len(detections); n_bg=max(0,n_events-n_true)
    times_bg = rng.uniform(0, time_span_seconds, size=n_bg)
    ras_bg = rng.uniform(0,360,size=n_bg)
    u = rng.uniform(-1,1,size=n_bg); decs_bg = np.rad2deg(np.arcsin(u))
    pos_err_bg = np.abs(rng.normal(params['pos_err'], params['pos_err']*0.5, size=n_bg))
    strengths_bg = np.abs(rng.normal(params['strength_mu'], params['strength_sigma'], size=n_bg))
    durations_bg = rng.exponential(5.0, size=n_bg)
    for i in range(n_bg):
        detections.append({"event_id":f"{m}_bg_{i:08d}","messenger":m,"time":float(times_bg[i]),"ra":float(ras_bg[i]),
                           "dec":float(decs_bg[i]),"pos_err":float(pos_err_bg[i]),"strength":float(strengths_bg[i]),
                           "duration":float(durations_bg[i]),"label":0,"source_id":None})
    df = pd.DataFrame(detections)
    return df.sample(frac=1.0, random_state=rng.randint(0,2**31-1)).reset_index(drop=True)

def generate_pairwise_features(df1, df2, m1, m2, out_csv_path, time_window, max_ang_sep, chunk_size, max_pairs_save, rng):
    ensure_dir(os.path.dirname(out_csv_path) or ".")
    df2s = df2.sort_values("time").reset_index(drop=True)
    times2 = df2s["time"].values; ra2=df2s["ra"].values; dec2=df2s["dec"].values; src2=df2s["source_id"].values
    header=False; pairs_count=0; n1=len(df1); iters=ceil(n1/chunk_size)
    for i in range(iters):
        start=i*chunk_size; end=min((i+1)*chunk_size,n1)
        block = df1.iloc[start:end]
        for _,e1 in block.iterrows():
            t=e1.time; left=np.searchsorted(times2,t-time_window); right=np.searchsorted(times2,t+time_window)
            if left>=right: continue
            idxs = np.arange(left,right)
            dtheta = angular_separation(e1.ra, e1.dec, ra2[idxs], dec2[idxs])
            mask = dtheta <= max_ang_sep
            if not mask.any(): continue
            for k in idxs[mask]:
                dt = float(times2[k]-t); dtheta_val=float(angular_separation(e1.ra,e1.dec,ra2[k],dec2[k]))
                strength_ratio = float(e1.strength/(df2s.iloc[k]['strength']+1e-12))
                label = 1 if (e1.source_id is not None and e1.source_id==src2[k]) else 0
                row = {"m1":m1,"m2":m2,"dt":dt,"dtheta":dtheta_val,"strength_ratio":strength_ratio,"label":int(label)}
                dfrow = pd.DataFrame([row])
                if not header:
                    dfrow.to_csv(out_csv_path, index=False, mode='w', header=True)
                    header=True
                else:
                    dfrow.to_csv(out_csv_path, index=False, mode='a', header=False)
                pairs_count+=1
                if max_pairs_save and pairs_count>=max_pairs_save:
                    return pairs_count
    return pairs_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_events_per_messenger", type=int, default=300000)
    parser.add_argument("--n_true_sources", type=int, default=50000)
    parser.add_argument("--time_span_days", type=float, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="synthetic_multimessenger_data")
    parser.add_argument("--max_pairs_per_pair", type=int, default=200000)
    parser.add_argument("--time_window_seconds", type=int, default=300)
    parser.add_argument("--max_angular_sep_deg", type=float, default=20.0)
    parser.add_argument("--chunk_size", type=int, default=20000)
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    if args.test_mode:
        args.n_events_per_messenger = 10000
        args.n_true_sources = 2000
        args.max_pairs_per_pair = 20000
        args.chunk_size = 2000

    rng = np.random.RandomState(args.seed)
    time_span_seconds = args.time_span_days * 24 * 3600
    ensure_dir(args.out_dir)

    print("Simulating true sources...")
    df_sources = simulate_true_sources(args.n_true_sources, time_span_seconds, rng)
    df_sources.to_csv(os.path.join(args.out_dir,"true_sources.csv"), index=False)

    messenger_dfs = {}
    for m in MESSENGERS:
        print(f"Simulating {m} ...")
        dfm = simulate_messenger(m, args.n_events_per_messenger, df_sources, time_span_seconds, rng, MESS_PARAMS[m])
        save_csv(dfm, os.path.join(args.out_dir,f"{m}_detections.csv"))
        messenger_dfs[m]=dfm
    # plots
    plot_histograms([messenger_dfs[m] for m in MESSENGERS], MESSENGERS, "strength", os.path.join(args.out_dir,"strength_distributions.png"), "Strengths")
    plot_histograms([messenger_dfs[m] for m in MESSENGERS], MESSENGERS, "pos_err", os.path.join(args.out_dir,"pos_err_distributions.png"), "Pos Err")

    # pairwise
    pair_results={}
    for (m1,m2) in combinations(MESSENGERS,2):
        print(f"Generating pairs {m1}-{m2} ...")
        if len(messenger_dfs[m1])<=len(messenger_dfs[m2]):
            small, big = messenger_dfs[m1], messenger_dfs[m2]; sm, bg = m1, m2
        else:
            small, big = messenger_dfs[m2], messenger_dfs[m1]; sm, bg = m2, m1
        out_csv = os.path.join(args.out_dir, f"pairs_{sm}_{bg}.csv")
        cnt = generate_pairwise_features(small,big,sm,bg,out_csv,args.time_window_seconds,args.max_angular_sep_deg,args.chunk_size,args.max_pairs_per_pair,rng)
        pair_results[f"{m1}-{m2}"]=int(cnt)
        print(f"Saved {cnt} pairs for {m1}-{m2}")
    with open(os.path.join(args.out_dir,"generation_metadata.json"),"w") as fh:
        json.dump({"config":vars(args),"pair_counts":pair_results},fh,indent=2)
    print("Done. Outputs in", args.out_dir)

if __name__=='__main__': main()
