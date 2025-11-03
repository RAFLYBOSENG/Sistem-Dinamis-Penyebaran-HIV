import os
import uuid
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# lokasi file history
HISTORY_FILE = os.path.join("data", "history.csv")
os.makedirs("data", exist_ok=True)


# -----------------------------
# Control signal generator
# -----------------------------
def control_signal(t, signal_type="none", amp=0.5, freq=1.0, step_time=None):
    """
    Menghasilkan faktor skalar u(t) yang mengalikan beta:
      beta_eff = beta * u(t)

    - t: numpy array (waktu)
    - signal_type: "none","step","impulse","ramp","sin"
    - amp: amplitude relative change (mis. 0.5 -> +50%)
    - freq: frequency (untuk sin)
    - step_time: waktu mulai step (jika None dipilih tengah t)
    """
    t = np.array(t)
    if signal_type == "step":
        if step_time is None:
            step_time = (t.max() - t.min()) / 2.0
        return np.where(t >= step_time, 1.0 + amp, 1.0)
    elif signal_type == "impulse":
        center = (t.max() + t.min()) / 2.0 if step_time is None else step_time
        width = max((t.max() - t.min()) * 0.02, 0.5)
        return 1.0 + amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    elif signal_type == "ramp":
        t0 = t.min()
        t1 = t.max()
        return 1.0 + amp * ((t - t0) / (t1 - t0))
    elif signal_type == "sin":
        # Normalisasi t to [0,1] then multiply freq
        return 1.0 + amp * np.sin(2 * np.pi * freq * (t - t.min()) / max(1.0, (t.max() - t.min())))
    else:
        return np.ones_like(t)


# -----------------------------
# Model SIR (time-varying beta via control signal)
# -----------------------------
def sir_ode(y, t, beta, gamma, N, signal_type="none", amp=0.5, freq=1.0, step_time=None):
    S, I, R = y
    # compute u(t) at scalar t (control_signal returns vector, so evaluate at t)
    u = control_signal(np.array(
        [t]), signal_type=signal_type, amp=amp, freq=freq, step_time=step_time)[0]
    beta_eff = beta * u
    dSdt = -beta_eff * S * I / N
    dIdt = beta_eff * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def simulate_sir(beta, gamma, N, I0, R0, days=100, npoints=500, signal_type="none", amp=0.5, freq=1.0, step_time=None):
    """
    Menjalankan simulasi SIR.
    Returns: t, S, I, R, u (control signal over t)
    """
    S0 = max(N - I0 - R0, 0.0)
    y0 = [S0, I0, R0]
    t = np.linspace(0, days, npoints)
    # solve ODE with odeint
    ret = odeint(sir_ode, y0, t, args=(
        beta, gamma, N, signal_type, amp, freq, step_time))
    S, I, R = ret.T
    u = control_signal(t, signal_type=signal_type, amp=amp,
                       freq=freq, step_time=step_time)
    return t, S, I, R, u


# -----------------------------
# Plotting (Plotly)
# -----------------------------
def plot_sir(t, S, I, R, u, signal_type, beta, gamma, N):
    # Buat subplot dengan 2 baris: 1 untuk SIR, 1 untuk kontrol
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],  # plot atas lebih besar
        vertical_spacing=0.1,
        subplot_titles=(
            f'Populasi HIV (S, I, R)',
            f'Kontrol Sinyal β(t) - Tipe: {signal_type.upper()}'
        ),
        shared_xaxes=True
    )
    
    # Plot 1: Populasi S, I, R
    fig.add_trace(
        go.Scatter(x=t, y=S, mode="lines",
                  name="S (Susceptible)", 
                  line=dict(color="blue", width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=I, mode="lines",
                  name="I (Infected)", 
                  line=dict(color="red", width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=R, mode="lines",
                  name="R (Recovered)", 
                  line=dict(color="green", width=2)),
        row=1, col=1
    )
    
    # Plot 2: Kontrol signal
    fig.add_trace(
        go.Scatter(x=t, y=u, mode="lines", 
                  name=f"Sinyal β(t)", 
                  line=dict(color="orange", width=2, dash="dash")),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Waktu (hari)", row=2, col=1)
    fig.update_yaxes(title_text="Jumlah Populasi", row=1, col=1)
    fig.update_yaxes(title_text="Faktor β(t)", row=2, col=1,
                    range=[max(0.0, u.min()*0.95), u.max()*1.05])
    
    fig.update_layout(
        title_text=f"Simulasi Dinamik HIV (Model SIR) - Parameter: β={beta:.3f}, γ={gamma:.3f}, N={int(N)}",
        title_x=0.5,
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


# -----------------------------
# History management
# -----------------------------
def _ensure_history():
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=["id", "timestamp", "beta", "gamma", "N", "I0", "R0", "days",
                                   "signal", "amp", "freq", "step_time", "peak_I", "peak_day", "final_S", "final_I", "final_R", "summary"])
        df.to_csv(HISTORY_FILE, index=False)
    else:
        # Jika file ada tapi kosong atau hanya header, buat ulang dengan header
        try:
            stat = os.stat(HISTORY_FILE)
            # Jika file sangat kecil (< 200 bytes), kemungkinan hanya header
            if stat.st_size < 200:
                with open(HISTORY_FILE, 'r') as f:
                    lines = f.readlines()
                # Jika hanya 1 baris atau kurang, anggap hanya header
                if len(lines) <= 1:
                    df = pd.DataFrame(columns=["id", "timestamp", "beta", "gamma", "N", "I0", "R0", "days",
                                               "signal", "amp", "freq", "step_time", "peak_I", "peak_day", "final_S", "final_I", "final_R", "summary"])
                    df.to_csv(HISTORY_FILE, index=False)
        except Exception:
            pass


def save_history_entry(beta, gamma, N, I0, R0, days, signal_type, amp=0.5, freq=1.0, step_time=None, summary_text=None, peak_I=None, peak_day=None, final_S=None, final_I=None, final_R=None):
    """
    Tambahkan satu baris ke history.csv dengan id unik.
    """
    _ensure_history()
    df = pd.read_csv(HISTORY_FILE)
    # generate id based on uuid to avoid collisions
    entry_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()
    record = {
        "id": entry_id,
        "timestamp": ts,
        "beta": beta,
        "gamma": gamma,
        "N": N,
        "I0": I0,
        "R0": R0,
        "days": days,
        "signal": signal_type,
        "amp": amp,
        "freq": freq,
        "step_time": step_time if step_time is not None else "",
        "peak_I": peak_I,
        "peak_day": peak_day,
        "final_S": final_S,
        "final_I": final_I,
        "final_R": final_R,
        "summary": summary_text if summary_text is not None else ""
    }
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)
    return entry_id


def load_history_df():
    _ensure_history()
    df = pd.read_csv(HISTORY_FILE)
    # Jika file kosong atau hanya header, return DataFrame kosong dengan kolom yang benar
    if df.empty:
        return pd.DataFrame(columns=["id", "timestamp", "beta", "gamma", "N", "I0", "R0", "days",
                                     "signal", "amp", "freq", "step_time", "peak_I", "peak_day", "final_S", "final_I", "final_R", "summary"])
    return df


def get_history_entry(entry_id):
    df = load_history_df()
    row = df[df["id"] == entry_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# -----------------------------
# Summary generator (interpretasi otomatis)
# -----------------------------
def generate_summary(beta, gamma, N, I0, R0, days, t, S, I, R, u, signal_type):
    """
    Buat ringkasan deskriptif otomatis berdasarkan output simulasi.
    Mengembalikan text summary dan beberapa statistik (peak, final).
    """
    # basic stats
    peak_idx = int(np.nanargmax(I))
    peak_I = float(I[peak_idx])
    peak_day = float(t[peak_idx])
    final_S = float(S[-1])
    final_I = float(I[-1])
    final_R = float(R[-1])

    R0_basic = beta / gamma if gamma != 0 else float("inf")

    # interpretasi
    lines = []
    lines.append(
        f"Model: SIR. Parameter: β={beta:.4g}, γ={gamma:.4g}, R₀ (β/γ) ≈ {R0_basic:.2f}.")
    lines.append(
        f"Populasi total N={int(N)}, kondisi awal I₀={int(I0)}, R₀={int(R0)}.")
    lines.append(
        f"Hasil utama: puncak jumlah terinfeksi ≈ {int(round(peak_I))} individu pada hari ke-{peak_day:.1f}.")
    lines.append(
        f"Pada akhir simulasi (t={int(days)} hari): S={int(round(final_S))}, I={int(round(final_I))}, R={int(round(final_R))}.")

    # qualitative interpretation based on R0
    if R0_basic > 1.5:
        lines.append(
            "Interpretasi: R₀ > 1 menunjukkan penyebaran yang cepat; wabah kemungkinan besar berkembang pesat tanpa intervensi.")
    elif 1.0 < R0_basic <= 1.5:
        lines.append(
            "Interpretasi: Penyebaran moderat; puncak infeksi lebih lambat dan lebih rendah.")
    else:
        lines.append(
            "Interpretasi: R₀ ≤ 1 menunjukkan infeksi cenderung menurun dan tidak menyebabkan wabah besar.")

    # effect of control signal
    if signal_type and signal_type != "none":
        # check how u changes: if u peaks >>1 means increase in beta
        mean_u = float(np.nanmean(u))
        max_u = float(np.nanmax(u))
        if max_u > 1.05:
            lines.append(
                f"Efek sinyal ({signal_type}): sinyal meningkatkan β sementara (max factor {max_u:.2f}), mempengaruhi kurva I(t).")
        else:
            lines.append(
                f"Efek sinyal ({signal_type}): perubahan β relatif kecil (rata-rata faktor {mean_u:.2f}).")
    else:
        lines.append(
            "Tidak ada sinyal kontrol: β konstan sepanjang simulasi (steady state).")

    # recommend action
    if R0_basic > 1.5:
        lines.append(
            "Saran: Pertimbangkan intervensi (reduksi kontak, terapi, atau pengobatan) untuk menurunkan β.")
    else:
        lines.append(
            "Saran: Kondisi relatif terkendali; ajukan pengamatan lanjutan untuk memastikan tren menurun.")

    summary_text = " ".join(lines)
    stats = {"peak_I": peak_I, "peak_day": peak_day,
             "final_S": final_S, "final_I": final_I, "final_R": final_R}
    return summary_text, stats


# -----------------------------
# High level helper: run + save
# -----------------------------
def run_simulation_and_save(beta, gamma, N, I0, R0, days=100, signal_type="none", amp=0.5, freq=1.0, step_time=None, npoints=500):
    t, S, I, R, u = simulate_sir(beta, gamma, N, I0, R0, days=days, npoints=npoints,
                                 signal_type=signal_type, amp=amp, freq=freq, step_time=step_time)
    summary_text, stats = generate_summary(
        beta, gamma, N, I0, R0, days, t, S, I, R, u, signal_type)
    # save to history
    entry_id = save_history_entry(beta=beta, gamma=gamma, N=N, I0=I0, R0=R0, days=days,
                                  signal_type=signal_type, amp=amp, freq=freq, step_time=step_time,
                                  summary_text=summary_text, peak_I=stats["peak_I"], peak_day=stats["peak_day"],
                                  final_S=stats["final_S"], final_I=stats["final_I"], final_R=stats["final_R"])
    plot_html = plot_sir(t, S, I, R, u, signal_type, beta, gamma, N)
    return {"id": entry_id, "t": t, "S": S, "I": I, "R": R, "u": u, "summary": summary_text, "plot_html": plot_html, "stats": stats}



