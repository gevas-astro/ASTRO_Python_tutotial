"""
swift_xrt.py
Helpers to download and plot Swift/XRT GRB light curves from the
UKSSDC repository (https://www.swift.ac.uk/xrt_curves/).
 
Summer School on Multimessenger Astronomy.
"""
 
import os
import re
from io import StringIO
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
BASE = "https://www.swift.ac.uk/xrt_curves"

c1 = 'cornflowerblue'
c2 = 'navy'
c3 = 'rebeccapurple'
c4 = '#CF6275'
c5 = 'maroon'



# --------------------------------------------------------------------------
# ID handling
# --------------------------------------------------------------------------
def target_id(obsid):
    """
    Turn anything the user pastes into the 8-digit XRT target ID used in the URL.
 
    Accepts: trigger number, target ID, or a full obsID.
      '1430212'      -> '01430212'   (zero-pad a bare trigger)
      '01430212'     -> '01430212'   (already a target ID)
      '01430212003'  -> '01430212'   (drop the 3-digit segment of an obsID)
    """
    s = "".join(ch for ch in str(obsid) if ch.isdigit())
    if len(s) > 8:          # full obsID = 8-digit target + 3-digit segment
        s = s[:8]
    return s.zfill(8)



# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def _parse_meta(text):
    """Pull (name, trigger) out of the header comment line."""
    name, trig = None, None
    for line in text.splitlines():
        if "name:" in line:
            m = re.search(r"trigger:\s*(\d+)", line)
            if m:
                trig = m.group(1)
            m = re.search(r"name:\s*(.+?)\s*$", line)
            if m:
                name = m.group(1).strip()
            break
    return name, trig


def _parse_qdp(text):
    """
    Robustly parse a Swift QDP light curve into a 6-column array:
        t, t+, t-, f, f+, f-
    Skips comments (!), the 'READ TERR' directive and 'NO ...' block separators,
    so it does NOT depend on a hard-coded skiprows.
    """
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("!") or s.upper().startswith("READ"):
            continue
        if s.upper().startswith("NO"):
            continue
        parts = s.split()
        try:
            vals = [float(p) for p in parts[:6]]
        except ValueError:
            continue
        if len(vals) == 6:
            rows.append(vals)
    return np.array(rows)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def get_xrt(obsid, kind="flux", datadir=".", save=True, timeout=30):
    """
    Download a Swift/XRT light curve and split it into detections / upper limits.

    Parameters
    ----------
    obsid : str/int   trigger, target ID or full obsID (see target_id()).
    kind  : 'flux' -> flux.qdp  (erg/cm^2/s),  'rate' -> curve.qdp (count/s).
    datadir : where to save the raw file.
    save  : write the raw .qdp next to the notebook.

    Returns
    -------
    data, data_det, data_ul, meta
        meta is a dict: {'name', 'trigger', 'target_id', 'kind', 'url'}.
    """
    tid = target_id(obsid)
    fname = {"flux": "flux.qdp", "rate": "curve.qdp"}[kind]
    url = f"{BASE}/{tid}/{fname}"

    r = requests.get(url, allow_redirects=True, timeout=timeout)
    r.raise_for_status()
    text = r.text

    if save:
        os.makedirs(datadir, exist_ok=True)
        with open(os.path.join(datadir, f"{tid}_{kind}.qdp"), "w") as fh:
            fh.write(text)

    name, trig = _parse_meta(text)
    arr = _parse_qdp(text)
    if arr.size == 0:
        raise ValueError(f"No numeric data parsed from {url}")

    data = pd.DataFrame(arr, columns=["t", "tu", "tl", "f", "fu", "fl"])
    det = data[data["fu"] > 0.0]
    ul = data[data["fu"] == 0.0]

    meta = {"name": name, "trigger": trig, "target_id": tid,
            "kind": kind, "url": url}
    return data, det, ul, meta


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot_xrt(det, ul, meta, ax=None):
    """Plot detections (with x/y errors) and upper limits on log-log axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(det["t"], det["f"],
                xerr=[-det["tl"], det["tu"]],
                yerr=[-det["fl"], det["fu"]],
                marker="o", markersize=4, ls=" ", color=c1, label="detections")

    if len(ul) > 0:
        ax.errorbar(ul["t"], ul["f"],
                    xerr=[-ul["tl"], ul["tu"]],
                    yerr=ul["f"] / 2.0, uplims=True,
                    ls=" ", color=c4, label="upper limits")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time since trigger (s)")
    if meta["kind"] == "flux":
        ax.set_ylabel(r"Flux 0.3$-$10 keV (erg cm$^{-2}$ s$^{-1}$)")
    else:
        ax.set_ylabel("Swift/XRT rate (c/s)")

    title = meta["name"] or f"target {meta['target_id']}"
    ax.set_title(f"XRT light curve: {title}")
    if len(ul) > 0:
        ax.legend()
    return ax


# --------------------------------------------------------------------------
# One-call convenience: download + plot
# --------------------------------------------------------------------------
def show_grb(obsid, kind="flux", datadir=".", save=True):
    """Download and immediately plot. Returns (data, det, ul, meta)."""
    data, det, ul, meta = get_xrt(obsid, kind=kind, datadir=datadir, save=save)
    plot_xrt(det, ul, meta)
    print(f"{meta['name']}  (trigger {meta['trigger']}, target {meta['target_id']})")
    print(f"  {len(det)} detections, {len(ul)} upper limits  <- {meta['url']}")
    return data, det, ul, meta


# --------------------------------------------------------------------------
# Browse the repository: name <-> target ID
# --------------------------------------------------------------------------
def list_recent(full=False, timeout=30):
    """
    Scrape the repository index for (GRB name -> target ID).

    full=False : the 'Recently observed GRBs' box on the front page.
    full=True  : every light curve (allcurves.php) -- a long list.

    Returns a DataFrame with columns ['name', 'target_id', 'url'].
    """
    url = f"{BASE}/allcurves.php" if full else f"{BASE}/"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    # links look like  href="/xrt_curves/03000405/"  with the GRB name as text
    pat = re.compile(r'href="/xrt_curves/(\d{8})/"[^>]*>\s*(GRB[^<]+?)\s*<', re.I)
    seen, rows = set(), []
    for tid, name in pat.findall(r.text):
        if tid in seen:
            continue
        seen.add(tid)
        rows.append({"name": name.strip(), "target_id": tid,
                     "url": f"{BASE}/{tid}/"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# GRB catalogue: SPER prompt-position table -> local CSV
# --------------------------------------------------------------------------
SPER_URL = "https://www.swift.ac.uk/sper/"


def update_grb_list(csv_path="swift_grb_list.csv", url=SPER_URL,
                    save=True, timeout=60):
    """
    Fetch the Swift-XRT SPER prompt-position table and save it locally as CSV.

    This is the master GRB list: every BAT trigger with its presumed name,
    position, position error, UVOT-enhanced flag and XRT redshift limit.
    The trigger number maps directly to the light-curve target ID, so any
    row feeds straight into show_grb(row['trigger']).

    Returns a DataFrame with columns:
        trigger, target_id, name, ra, dec, pos_err_arcsec,
        enhanced, z_limit, has_position, curve_url
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    df = None
    for t in pd.read_html(StringIO(r.text)):
        cols = " ".join(str(c).lower() for c in t.columns)
        if "trigger" in cols and "name" in cols:
            df = t
            break
    if df is None:
        raise ValueError("Could not find the SPER positions table on the page.")

    # 7 columns in a fixed order -> rename by position (robust to header wording)
    std = ["trigger", "name", "ra", "dec", "pos_err_arcsec", "enhanced", "z_limit"]
    df = df.iloc[:, :7].copy()
    df.columns = std

    # rows with no localisation carry "No source found..." across the columns
    nosrc = df["ra"].astype(str).str.contains("No source", case=False, na=False)
    df.loc[nosrc, ["ra", "dec", "pos_err_arcsec", "enhanced", "z_limit"]] = pd.NA
    df["has_position"] = ~nosrc

    df["trigger"] = df["trigger"].astype(str).str.extract(r"(\d+)")[0]
    df["target_id"] = df["trigger"].apply(target_id)
    df["pos_err_arcsec"] = pd.to_numeric(df["pos_err_arcsec"], errors="coerce")
    z = (df["z_limit"].astype(str)
         .str.replace("<", "", regex=False).str.strip())
    df["z_limit"] = pd.to_numeric(z, errors="coerce")
    df["curve_url"] = BASE + "/" + df["target_id"] + "/"

    df = df[["trigger", "target_id", "name", "ra", "dec", "pos_err_arcsec",
             "enhanced", "z_limit", "has_position", "curve_url"]]

    if save:
        df.to_csv(csv_path, index=False)
        print(f"saved {len(df)} GRBs -> {csv_path}")
    return df