from flask import Flask, request, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    df = pd.read_excel(file)

    anomalies, plot_url, fp_rate = detect_anomalies(df)
    return render_template('results.html', tables=[anomalies.to_html(classes='data')], titles=anomalies.columns.values, plot_url=plot_url, fp_rate=fp_rate)

@app.route('/plot.png')
def plot_png():
    plot = plt.gcf()
    output = io.BytesIO()
    plot.savefig(output, format='png')
    output.seek(0)
    return send_file(output, mimetype='image/png')

def detect_anomalies(df):
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    df['predicted_anomaly'] = model.fit_predict(df.select_dtypes(include=[float, int]))

    anomalies = df[df['predicted_anomaly'] == -1]

    false_positives = df[(df['predicted_anomaly'] == -1) & (df['ground_truth'] == 0)]
    total_actual_negatives = len(df[df['ground_truth'] == 0])
    fp_rate = len(false_positives) / total_actual_negatives if total_actual_negatives > 0 else 0

    plt.figure(figsize=(10, 6))
    for column in df.select_dtypes(include=[float, int]).columns:
        plt.plot(df.index, df[column], label=column)
    plt.scatter(anomalies.index, anomalies[df.select_dtypes(include=[float, int]).columns[0]], color='red', label='Anomalies')
    plt.legend()
    plt.title('Anomaly Detection')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig('plot.png')

    return anomalies, '/plot.png', fp_rate

if __name__ == '__main__':
    app.run(debug=True)
