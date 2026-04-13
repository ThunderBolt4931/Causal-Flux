import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator

def plot_sentiment_curves(results, id_to_transcript):
    top_5 = results[:10]
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("husl", len(top_5))
    for idx, item in enumerate(top_5):
        tid = item['transcript_id']
        transcript_data = id_to_transcript.get(tid)
        if not transcript_data or 'turns' not in transcript_data:
            continue
        sentiments = [t.get('sentiment', {}).get('score', 0) for t in transcript_data['turns']]
        turns = np.array(range(1, len(sentiments) + 1))
        scores = np.array(sentiments)
        if len(turns) > 3:
            try:
                x_new = np.linspace(turns.min(), turns.max(), 300)
                spl = make_interp_spline(turns, scores, k=3)
                y_smooth = spl(x_new)
                plt.plot(x_new, y_smooth, label=f"Rank {item['rank']}: {tid[:8]}...", color=colors[idx], linewidth=2.5, alpha=0.8)
                plt.scatter(turns, scores, color=colors[idx], s=30)
            except Exception:
                plt.plot(turns, scores, label=f"Rank {item['rank']}: {tid[:8]}...", color=colors[idx], linewidth=2.5, alpha=0.8)
        else:
            plt.plot(turns, scores, label=f"Rank {item['rank']}: {tid[:8]}...", color=colors[idx], linewidth=2.5, alpha=0.8)
    plt.title("Sentiment Progression (Top 10 Retrieved Calls)", fontsize=16, pad=20)
    plt.xlabel("Conversation Turn", fontsize=12)
    plt.ylabel("Sentiment Score", fontsize=12)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()

def plot_average_sentiment_trend(results, id_to_transcript):
    top_k = results[:10]
    all_series = []
    for item in top_k:
        tid = item['transcript_id']
        transcript_data = id_to_transcript.get(tid)
        if not transcript_data or 'turns' not in transcript_data:
            continue
        sentiments = [t.get('sentiment', {}).get('score', 0) for t in transcript_data['turns']]
        all_series.append(sentiments)
    if not all_series:
        return
    max_len = max(len(s) for s in all_series)
    avg_scores = []
    turn_indices = []
    for i in range(max_len):
        turn_values = []
        for series in all_series:
            if i < len(series):
                turn_values.append(series[i])
        if turn_values:
            avg_scores.append(np.mean(turn_values))
            turn_indices.append(i + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    turns = np.array(turn_indices)
    scores = np.array(avg_scores)
    
    if len(turns) > 3:
        try:
            x_new = np.linspace(turns.min(), turns.max(), 300)
            spl = make_interp_spline(turns, scores, k=3)
            y_smooth = spl(x_new)
            ax.plot(x_new, y_smooth, label="Average Sentiment Trend", color="#2c3e50", linewidth=3)
            ax.fill_between(x_new, y_smooth, 0, where=(y_smooth>=0), interpolate=True, color='blue', alpha=0.3)
            ax.fill_between(x_new, y_smooth, 0, where=(y_smooth<=0), interpolate=True, color='red', alpha=0.3)
        except Exception:
             ax.plot(turns, scores, label="Average Sentiment Trend", color="#2c3e50", linewidth=3)
             ax.fill_between(turns, scores, 0, where=(scores>=0), interpolate=True, color='blue', alpha=0.3)
             ax.fill_between(turns, scores, 0, where=(scores<=0), interpolate=True, color='red', alpha=0.3)
    else:
        ax.plot(turns, scores, label="Average Sentiment Trend", color="#2c3e50", linewidth=3)
        ax.fill_between(turns, scores, 0, where=(scores>=0), interpolate=True, color='blue', alpha=0.3)
        ax.fill_between(turns, scores, 0, where=(scores<=0), interpolate=True, color='red', alpha=0.3)
    
    sc = ax.scatter(turns, scores, color='red', s=50, zorder=5, label="Top-k Calls Average Sentiment Score")
    
    annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        idx = ind["ind"][0]
        text = f"Turn: {int(turns[idx])}\nScore: {scores[idx]:.2f}"
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    ax.set_title("Aggregate Sentiment Trend (Average of Top 10 Calls)", fontsize=16, pad=20)
    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Average Sentiment Score", fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()