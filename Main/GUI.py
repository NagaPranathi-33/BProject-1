import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from Main import Run  # Make sure your Main.py file has a 'callmain' method

def run_model(dataset, tr_pct):
    """
    Calls your pipeline to compute metrics for all 5 models.
    Returns three lists: Accuracy, TPR, TNR each of length 5.
    """
    #A, Tpr, Tnr = Run.callmain(dataset, int(tr_pct))
    A, Tpr, Tnr = Run.callmain(dataset, int(tr_pct))
    return (
        A[0], A[1], A[2], A[3], A[4],
        Tpr[0], Tpr[1], Tpr[2], Tpr[3], Tpr[4],
        Tnr[0], Tnr[1], Tnr[2], Tnr[3], Tnr[4],
    )

def plot_graph(
    acc1, acc2, acc3, acc4, acc5,
    tpr1, tpr2, tpr3, tpr4, tpr5,
    tnr1, tnr2, tnr3, tnr4, tnr5
):
    # Convert inputs to floats
    acc = list(map(float, [acc1, acc2, acc3, acc4, acc5]))
    tpr = list(map(float, [tpr1, tpr2, tpr3, tpr4, tpr5]))
    tnr = list(map(float, [tnr1, tnr2, tnr3, tnr4, tnr5]))

    data = [acc, tpr, tnr]
    metrics = ['Accuracy', 'TPR', 'TNR']
    model_labels = ['Adaptive E-Bat+DBN', 'CBF+DBN', 'WOA+BRNN', 'Hybrid NN', 'Proposed SSPO-based DQN']

    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(dpi=120)
    for i in range(5):
        values = [data[j][i] for j in range(3)]
        bars = ax.bar(x + width*i - width*2, values, width, label=model_labels[i])
        # Optional: Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Value')
    ax.set_ylim(0, 1.05)
    ax.set_title('Performance Comparison of Classifiers')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()

    out_path = "comparison_plot.png"
    plt.savefig(out_path)
    plt.close()
    return out_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📊 Big Data Classifier Performance")

    with gr.Row():
        dataset_input = gr.Dropdown(
            choices=["Adult", "Credit_Approval", "adult_UCI", "creditcard", "kddcup99", "transfers_sorted", "7817_1_cleaned"],
            label="Select Dataset"
        )
        tr_pct_input = gr.Slider(
            minimum=10, maximum=100, step=10, value=80,
            label="Training Data (%)"
        )
        run_btn = gr.Button("▶️ Run Models")

    with gr.Row():
        acc1 = gr.Textbox(label="Accuracy: Adaptive E-Bat+DBN")
        acc2 = gr.Textbox(label="Accuracy: CBF+DBN")
        acc3 = gr.Textbox(label="Accuracy: WOA+BRNN")
        acc4 = gr.Textbox(label="Accuracy: Hybrid NN")
        acc5 = gr.Textbox(label="Accuracy: SSPO-DQN")

    with gr.Row():
        tpr1 = gr.Textbox(label="TPR: Adaptive E-Bat+DBN")
        tpr2 = gr.Textbox(label="TPR: CBF+DBN")
        tpr3 = gr.Textbox(label="TPR: WOA+BRNN")
        tpr4 = gr.Textbox(label="TPR: Hybrid NN")
        tpr5 = gr.Textbox(label="TPR: SSPO-DQN")

    with gr.Row():
        tnr1 = gr.Textbox(label="TNR: Adaptive E-Bat+DBN")
        tnr2 = gr.Textbox(label="TNR: CBF+DBN")
        tnr3 = gr.Textbox(label="TNR: WOA+BRNN")
        tnr4 = gr.Textbox(label="TNR: Hybrid NN")
        tnr5 = gr.Textbox(label="TNR: SSPO-DQN")

    graph_btn = gr.Button("📈 Show Comparison Graph")
    graph_img = gr.Image(label="Performance Chart")

    run_btn.click(
        fn=run_model,
        inputs=[dataset_input, tr_pct_input],
        outputs=[acc1, acc2, acc3, acc4, acc5,
                 tpr1, tpr2, tpr3, tpr4, tpr5,
                 tnr1, tnr2, tnr3, tnr4, tnr5]
    )

    graph_btn.click(
        fn=plot_graph,
        inputs=[acc1, acc2, acc3, acc4, acc5,
                tpr1, tpr2, tpr3, tpr4, tpr5,
                tnr1, tnr2, tnr3, tnr4, tnr5],
        outputs=graph_img
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
