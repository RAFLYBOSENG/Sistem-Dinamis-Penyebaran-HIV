from flask import Flask, render_template, request, redirect, url_for, flash
from train import run_simulation_and_save, load_history_df, get_history_entry, simulate_sir, plot_sir
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-secret"

# -------------------------
# Home: input form
# -------------------------


@app.route("/", methods=["GET", "POST"])
def index():
    # If rerun parameters passed via query args, prefill form
    prefill = {}
    if request.method == "GET":
        prefill = {
            "beta": request.args.get("beta", "0.3"),
            "gamma": request.args.get("gamma", "0.1"),
            "N": request.args.get("N", "1000000"),
            "I0": request.args.get("I0", "10"),
            "R0": request.args.get("R0", "0"),
            "days": request.args.get("days", "100"),
            "signal_type": request.args.get("signal", "none")
        }

    if request.method == "POST":
        try:
            beta = float(request.form.get("beta", 0.3))
            gamma = float(request.form.get("gamma", 0.1))
            N = float(request.form.get("N", 1_000_000))
            I0 = float(request.form.get("I0", 1.0))
            R0 = float(request.form.get("R0", 0.0))
            days = int(request.form.get("days", 100))
            signal_type = request.form.get("signal_type", "none")
            amp = float(request.form.get("amp", 0.5))
            freq = float(request.form.get("freq", 1.0))
            step_time_val = request.form.get("step_time", "")
            step_time = float(step_time_val) if step_time_val else None

            # run sim + save history
            result = run_simulation_and_save(beta=beta, gamma=gamma, N=N, I0=I0, R0=R0,
                                             days=days, signal_type=signal_type, amp=amp, freq=freq, step_time=step_time)
            entry_id = result["id"]
            return redirect(url_for("history_detail", entry_id=entry_id))

        except Exception as e:
            flash(
                f"Terjadi kesalahan saat menjalankan simulasi: {e}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html", prefill=prefill)


# -------------------------
# History list
# -------------------------
@app.route("/history")
def history():
    df = load_history_df()

    # Pastikan kolom timestamp ada sebelum sort
    if "timestamp" not in df.columns or df.empty:
        df = pd.DataFrame(
            columns=["id", "timestamp", "beta", "gamma", "N", "I0", "R0", "days", "signal"])
        return render_template("history.html", table=[])

    # Hapus baris dengan timestamp kosong/NaN
    df = df.dropna(subset=["timestamp"])

    # Pastikan masih ada data setelah hapus NaN
    if df.empty:
        return render_template("history.html", table=[])

    # Sort terbaru dulu
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # convert timestamp to readable
    if "timestamp" in df.columns:
        df["timestamp_readable"] = pd.to_datetime(
            df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return render_template("history.html", table=df.to_dict(orient="records"))


# -------------------------
# History detail (view summary + plot + rerun)
# -------------------------
@app.route("/history/<entry_id>")
def history_detail(entry_id):
    entry = get_history_entry(entry_id)
    if not entry:
        flash("Riwayat tidak ditemukan.", "warning")
        return redirect(url_for("history"))

    # regenerate plot from parameters (so plot interactive always available)
    beta = float(entry["beta"])
    gamma = float(entry["gamma"])
    N = float(entry["N"])
    I0 = float(entry["I0"])
    R0 = float(entry["R0"])
    days = int(entry["days"])
    signal = entry.get("signal", "none")
    amp = float(entry.get("amp", 0.5)) if entry.get("amp", "") != "" else 0.5
    freq = float(entry.get("freq", 1.0)) if entry.get(
        "freq", "") != "" else 1.0
    step_time = float(entry.get("step_time")) if entry.get(
        "step_time", "") != "" else None

    t, S, I, R, u = simulate_sir(beta=beta, gamma=gamma, N=N, I0=I0, R0=R0,
                                 days=days, signal_type=signal, amp=amp, freq=freq, step_time=step_time)
    plot_html = plot_sir(t, S, I, R, u, signal, beta, gamma, N)

    # prepare entry dict for template
    display_entry = {
        "id": entry["id"],
        "timestamp": pd.to_datetime(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
        "beta": beta,
        "gamma": gamma,
        "N": int(float(N)),
        "I0": int(float(I0)),
        "R0": int(float(R0)),
        "days": int(days),
        "signal": signal,
        "amp": amp,
        "freq": freq,
        "step_time": step_time,
        "peak_I": int(float(entry.get("peak_I"))) if entry.get("peak_I", "") != "" else None,
        "peak_day": float(entry.get("peak_day")) if entry.get("peak_day", "") != "" else None,
        "final_S": int(float(entry.get("final_S"))) if entry.get("final_S", "") != "" else None,
        "final_I": int(float(entry.get("final_I"))) if entry.get("final_I", "") != "" else None,
        "final_R": int(float(entry.get("final_R"))) if entry.get("final_R", "") != "" else None,
        "summary": entry.get("summary", "")
    }

    return render_template("detail.html", entry=display_entry, plot_html=plot_html)


# -------------------------
# Rerun route: prefill index form with params from history
# -------------------------
@app.route("/rerun/<entry_id>")
def rerun(entry_id):
    entry = get_history_entry(entry_id)
    if not entry:
        flash("Riwayat tidak ditemukan.", "warning")
        return redirect(url_for("history"))
    # redirect to index with query params to prefill
    params = {
        "beta": entry.get("beta"),
        "gamma": entry.get("gamma"),
        "N": entry.get("N"),
        "I0": entry.get("I0"),
        "R0": entry.get("R0"),
        "days": entry.get("days"),
        "signal": entry.get("signal")
    }
    return redirect(url_for("index", **params))


if __name__ == "__main__":
    # ensure history file exists
    try:
        load_history_df()
    except Exception:
        pass
    app.run(debug=True, port=5000)
