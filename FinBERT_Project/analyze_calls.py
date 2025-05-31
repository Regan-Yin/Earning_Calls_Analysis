#!/usr/bin/env python3
# analyze_calls.py
"""
ISOM 3350 Final Project
Main part of the project:
Analyze financial calls for sentiment and price change
Author: Regan Yin
Date: 2025-05-11

Full pipeline:
 1) Transcript → per-sentence FinBERT-tone sentiment
 2) Compute transcript-level sentiment index
 3) Fetch nearest trading-day closes → price_change_pct & price_up
 4) Classification: LogisticRegression (sentiment_index → price_up)
 5) Event Study: compute Abnormal Returns vs S&P500 benchmark, CAR[-1,+1]
 6) Granger Causality Test between sentiment_index & price_change_pct

Outputs under ./report:
  • details_<key>.csv
  • sentiment_summary.csv
  • price_change.csv
  • classification_report.txt
  • event_study.csv
  • event_study_plot.jpg
  • granger.txt
  • corr_heatmap.jpg
  • scatter_with_reg.jpg
"""

import os, json, warnings, time
from datetime import timedelta
import pandas as pd, numpy as np
import torch, yfinance as yf
import seaborn as sns, matplotlib.pyplot as plt
from transformers import (
    BertTokenizer, BertForSequenceClassification, pipeline
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr, spearmanr, kendalltau

# ─── CONFIG ───────────────────────────────────────────────────────────────────

TRANSCRIPT_DIR = "./transcript"
METADATA_PATH  = "./metadata.json"
REPORT_DIR     = "./report"
THRESHOLD      = 0.9   # confidence for including sentences in index
EVENT_WINDOW   = 1     # days before/after for event study & price_change
SLEEP_SECONDS  = 60     # wait after batch download, in order to avoid ban by yfinance

warnings.filterwarnings("ignore")
os.makedirs(REPORT_DIR, exist_ok=True)

# ─── LOAD METADATA & MODEL ────────────────────────────────────────────────────
# 1) Load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# 2) Load FinBERT-tone
finbert   = BertForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone", num_labels=3
)
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
device    = "cuda" if torch.cuda.is_available() else "cpu"
nlp       = pipeline(
    "text-classification",
    model=finbert,
    tokenizer=tokenizer,
    device=device,
    batch_size=16
)
def normalize_label(lbl): return lbl.lower()

# ─── TEXT PROCESSING ──────────────────────────────────────────────────────────

def extract_sentences(path):
    data = json.load(open(path,"r",encoding="utf-8"))
    recs = []
    for i, seg in enumerate(data.get("transcription",[]),1):
        txt = seg.get("text","").strip()
        if txt: recs.append({"index":i,"sentence":txt})
    return pd.DataFrame(recs)

def analyze_sentences(df):
    outs = nlp(df["sentence"].tolist())
    rows=[]
    for i,out in enumerate(outs):
        rows.append({
            "index":    int(df.loc[i,"index"]),
            "sentence": df.loc[i,"sentence"],
            "label":    normalize_label(out["label"]),
            "score":    float(out["score"])
        })
    return pd.DataFrame(rows)

def compute_overall(df):
    pos = df.loc[(df.label=="positive")&(df.score>=THRESHOLD),"score"].sum()
    neg = df.loc[(df.label=="negative")&(df.score>=THRESHOLD),"score"].sum()
    idx = 0.0 if pos+neg==0 else (pos-neg)/(pos+neg)
    lbl = "positive" if pos>neg else "negative" if neg>pos else "neutral"
    return lbl, idx

# ─── BATCH PRICE RETRIEVAL ────────────────────────────────────────────────────

def batch_price_for_ticker(tkr, event_dates, window=EVENT_WINDOW):
    """Batch download all event window daily bars for the ticker, 
    and return the close prices before/after each event date.
    In order to avoid access restriction from yFinance"""
    start = min(event_dates) - timedelta(days=window)
    end   = max(event_dates) + timedelta(days=window) + timedelta(days=1)
    df = yf.download(tkr, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=False)
    df.index = df.index.date

    out = {}
    alld = sorted(df.index)
    for ed in event_dates:
        prior = max((d for d in alld if d < ed), default=None)
        after = min((d for d in alld if d > ed), default=None)
        if prior and after:
            out[ed] = (
                df.loc[prior,"Close"],
                df.loc[after,"Close"],
                prior,
                after
            )
        else:
            out[ed] = (None,None,prior,after)
    return out

def compute_price_change(df_overall):
    recs = []
    for tkr, grp in df_overall.groupby("ticker"):
        edates = grp["event_date"].apply(pd.to_datetime).dt.date.tolist()
        print(f"→ Batch downloading {tkr} price，event number={len(edates)} …")
        price_map = batch_price_for_ticker(tkr, edates)
        print(f"  {tkr} Download completed，sleep {SLEEP_SECONDS}s …")
        time.sleep(SLEEP_SECONDS)
        for _, r in grp.iterrows():
            ed = pd.to_datetime(r["event_date"]).date()
            c0,c1,d0,d1 = price_map.get(ed,(None,None,None,None))
            if c0 is None or c1 is None: continue
            pct = (c1-c0)/c0
            recs.append({
                "transcript":        r["transcript"],
                "ticker":            tkr,
                "event_date":        ed,
                "sentiment_index":   r["sentiment_index"],
                "close_before":      float(c0),
                "close_after":       float(c1),
                "close_before_date": d0,
                "close_after_date":  d1,
                "price_change_pct":  float(pct),
                "price_up":          int(pct>0),
            })
    return pd.DataFrame(recs)

# ─── CLASSIFICATION ────────────────────────────────────────────────────────────

def classification_analysis(df_prices):
    X = df_prices[["sentiment_index"]]
    y = df_prices["price_up"]
    # check for at least two classes
    classes = y.unique()
    rpt = os.path.join(REPORT_DIR,"classification_report.txt")
    if len(classes)<2:
        # write a trivial report and skip fitting
        with open(rpt,"w") as f:
            f.write("Skipped — only one class present:\n")
            f.write(f"{classes.tolist()}, hit rate={(y==classes[0]).mean():.2%}\n")
        return None
    
    # otherwise train
    clf = LogisticRegression().fit(X,y)
    pred = clf.predict(X)
    with open(rpt,"w") as f:
        f.write("LogisticRegression(sentiment_index → price_up)\n")
        f.write(f"Accuracy :{accuracy_score(y,pred):.3f}\n"
                f"Precision:{precision_score(y,pred):.3f}\n"
                f"Recall   :{recall_score(y,pred):.3f}\n"
                f"F1       :{f1_score(y,pred):.3f}\n\n")
        f.write("Coefficients:\n")
        f.write(f"  Intercept:{clf.intercept_[0]:.4f}\n"
                f"  Coef     :{clf.coef_[0][0]:.4f}\n")
    return clf

# ─── EVENT STUDY ───────────────────────────────────────────────────────────────

def event_study(df_prices):
    # 1. Get benchmark data (^GSPC)
    dates = set(df_prices["close_before_date"]) | set(df_prices["close_after_date"])
    start, end = min(dates) - timedelta(days=2), max(dates) + timedelta(days=2)

    try:
        bench = yf.download("^GSPC", start=start.strftime("%Y-%m-%d"), 
                            end=(end+timedelta(1)).strftime("%Y-%m-%d"), 
                            auto_adjust=False, progress=False)
        bench.index = bench.index.date
    
    except Exception as e:
        print(f"⚠️ Failed to fetch benchmark (^GSPC): {e}")
        return

    if bench.empty:
        print("⚠️ Benchmark data is empty. Skipping event study.")
        return

    # 2. Compute abnormal returns
    recs=[]
    for _,r in df_prices.iterrows():
        d0,d1 = r["close_before_date"], r["close_after_date"]
        if d0 not in bench.index or d1 not in bench.index: continue
        ret_s = r["price_change_pct"]
        ret_b = (bench.loc[d1,"Close"]-bench.loc[d0,"Close"])/bench.loc[d0,"Close"]
        ab = ret_s - ret_b
        recs.append({
            **r.to_dict(),
            "benchmark_return":float(ret_b),
            "abnormal_return":float(ab)
        })
    df = pd.DataFrame(recs)
    if df.empty:
        print("⚠️ No available event study data, skipping...")
        return
    
    # 3. Compute CAR
    df["CAR"] = df["abnormal_return"].cumsum()  # cumulative abnormal return
    df.to_csv(os.path.join(REPORT_DIR,"event_study.csv"), index=False)

    # plot CAR by transcript
    plt.figure(figsize=(6,4))
    sns.barplot(x="transcript",y="CAR",data=df)
    plt.xticks(rotation=45,ha="right"); plt.title("Event Study: CAR")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR,"event_study_plot.jpg"))
    plt.close()

# ─── GRANGER CAUSALITY ────────────────────────────────────────────────────────

def granger_test(df_prices, maxlag=2):
    # Build 2-col frame
    df = df_prices[["price_change_pct","sentiment_index"]].reset_index(drop=True)

    # How many usable rows?
    n = len(df)
    if n <= maxlag + 1:
        with open(os.path.join(REPORT_DIR,"granger.txt"), "w") as f:
            f.write(f"Granger test skipped: only {n} observations, need > {maxlag+1}\n")
        return

    # OK to run
    res = grangercausalitytests(df, maxlag=maxlag, verbose=False)
    with open(os.path.join(REPORT_DIR,"granger.txt"), "w") as f:
        f.write("Granger Causality Test Results:\n")
        for lag in range(1, maxlag+1):
            pval = res[lag][0]["ssr_ftest"][1]
            f.write(f"  lag={lag:>2d}  p-value={pval:.4f}\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # A) Sentiment per transcript
    print("\n[Step A] Calculating sentiment for each transcript and generating sentiment_summary.csv...")
    summary=[]
    for fname in sorted(os.listdir(TRANSCRIPT_DIR)):
        if not fname.endswith(".json"): continue
        key = os.path.splitext(fname)[0]
        meta=metadata.get(key)
        if meta is None:
            print(f"⚠ skip {key}")
            continue
        path = os.path.join(TRANSCRIPT_DIR,fname)
        print(f"  Processing transcript: {key}")
        df_s = extract_sentences(path)
        df_d = analyze_sentences(df_s)
        lbl, idx = compute_overall(df_d)
        summary.append({
            "transcript":key,
            "ticker":meta["ticker"],
            "event_date":meta["event_date"],
            "sentiment_label":lbl,
            "sentiment_index":idx
        })
    df_over = pd.DataFrame(summary)
    df_over.to_csv(os.path.join(REPORT_DIR,"sentiment_summary.csv"), index=False)

    # B) Price change
    print("[Step B] Calculating price changes and generating price_change.csv...")
    df_pr = compute_price_change(df_over)
    df_pr.to_csv(os.path.join(REPORT_DIR,"price_change.csv"), index=False)

    # C) Classification
    print("[Step C] Running classification analysis and generating classification_report.txt...")
    clf = classification_analysis(df_pr)

    # D) Event Study
    print("[Step D] Performing event study and generating event_study.csv and event_study_plot.jpg...")
    event_study(df_pr)

    # E) Granger
    print("[Step E] Running Granger causality test and generating granger.txt...")
    granger_test(df_pr)

    # F) Correlation & Plots
    print("[Step F] Calculating correlations and generating corr_heatmap.jpg and scatter_with_reg.jpg...")
    si = df_pr["sentiment_index"]; pc = df_pr["price_change_pct"]
    pear_r,_ = pearsonr(si,pc) if len(si)>1 else (np.nan,np.nan)
    spr_rho,_= spearmanr(si,pc) if len(si)>1 else (np.nan,np.nan)
    corr_mat = pd.DataFrame({"sentiment_index":si,"price_change_pct":pc}).corr()
    # heatmap
    plt.figure(figsize=(4,4))
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", vmin=-1,vmax=1)
    plt.title(f"Pearson r={pear_r:.2f}, Spearman ρ={spr_rho:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR,"corr_heatmap.jpg"))
    plt.close()
    # scatter+reg
    plt.figure(figsize=(6,4))
    sns.regplot(x=si,y=pc,ci=None,line_kws={"color":"black"})
    plt.axhline(0,color="gray",ls="--"); plt.axvline(0,color="gray",ls="--")
    plt.xlabel("Sentiment Index"); plt.ylabel("Price Change %")
    plt.title("Sentiment vs Price Change")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR,"scatter_with_reg.jpg"))
    plt.close()

    print(f"\n✅ Reports under {REPORT_DIR}/:\n"
          "  sentiment_summary.csv\n"
          "  price_change.csv\n"
          "  classification_report.txt\n"
          "  event_study.csv + .jpg\n"
          "  granger.txt\n"
          "  corr_heatmap.jpg\n"
          "  scatter_with_reg.jpg\n")

if __name__=="__main__":
    main()
