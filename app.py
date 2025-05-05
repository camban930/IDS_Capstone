from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import time
import threading
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
socketio = SocketIO(app)

LOGS_FILE = "logs.csv"

def load_logs():
    df = pd.read_csv(LOGS_FILE)
    df["failed_attempt"] = df["status"].apply(lambda x: 1 if x == "failure" else 0)
    df["success_attempt"] = df["status"].apply(lambda x: 1 if x == "success" else 0)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def analyze_logins(df):
    ip_stats = df.groupby("source_ip").agg({
        "failed_attempt": ["sum", "count"],
        "success_attempt": "sum",
        "timestamp": ["min", "max"]
    }).reset_index()
    
    ip_stats.columns = ['source_ip', 'failed_sum', 'total_attempts', 'success_sum', 'first_attempt', 'last_attempt']
    
    ip_stats["suspicious"] = ip_stats["failed_sum"].apply(lambda x: 1 if x > 3 else 0)
    ip_stats["attempt_rate"] = ip_stats["total_attempts"] / ((ip_stats["last_attempt"] - ip_stats["first_attempt"]).dt.total_seconds() / 3600 + 1)
    return ip_stats

def generate_graphs(ip_stats):
    graphs = {}
    
    # Suspicious vs Normal IPs
    suspicious_ips = ip_stats[ip_stats["suspicious"] == 1]
    normal_ips = ip_stats[ip_stats["suspicious"] == 0]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Suspicious IPs", "Normal IPs"], 
           [len(suspicious_ips), len(normal_ips)], 
           color=["#ff6b6b", "#51cf66"])
    ax.set_title("Suspicious vs Normal IPs", pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    graphs['suspicious_chart'] = fig_to_base64(fig)
    
    # Attempts over time
    df = load_logs()
    time_series = df.set_index('timestamp').resample('5T').agg({
        'failed_attempt': 'sum',
        'success_attempt': 'sum'
    }).fillna(0)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_series.index, time_series['failed_attempt'], 
            label='Failed Attempts', color='#ff6b6b')
    ax.plot(time_series.index, time_series['success_attempt'], 
            label='Successful Attempts', color='#51cf66')
    ax.set_title("Login Attempts Over Time", pad=20)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    graphs['timeline_chart'] = fig_to_base64(fig)
    
    # Top suspicious IPs
    top_ips = ip_stats.sort_values('failed_sum', ascending=False).head(5)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_ips['source_ip'], top_ips['failed_sum'], 
            color=['#ff6b6b', '#ff8787', '#ffa8a8', '#ffc9c9', '#ffd8d8'])
    ax.set_title("Top Suspicious IPs", pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    graphs['top_ips_chart'] = fig_to_base64(fig)
    
    return graphs

def fig_to_base64(fig):
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def simulate_real_time_alerts(df):
    for index, row in df.iterrows():
        event = {
            "source_ip": row["source_ip"],
            "status": row["status"],
            "timestamp": row["timestamp"].strftime('%Y-%m-%d %H:%M:%S'),
            "severity": "high" if row["status"] == "failure" else "low"
        }
        socketio.emit("new_alert", event)
        time.sleep(2)

@app.route("/")
def dashboard():
    df = load_logs()
    ip_stats = analyze_logins(df)
    graphs = generate_graphs(ip_stats)
    stats = {
        'total_attempts': len(df),
        'failed_attempts': df['failed_attempt'].sum(),
        'suspicious_ips': len(ip_stats[ip_stats['suspicious'] == 1]),
        'unique_ips': len(ip_stats)
    }
    return render_template("dashboard.html", graphs=graphs, stats=stats)

@socketio.on("connect")
def handle_connect():
    df = load_logs()
    threading.Thread(target=simulate_real_time_alerts, args=(df[-20:],)).start()

if __name__ == "__main__":
    socketio.run(app, debug=True, host='127.0.0.1', port=55555, use_reloader=False)
