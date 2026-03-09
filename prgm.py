from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    df = pd.read_excel(file)
    anomalies = detect_anomalies(df)
    return render_template('results.html', tables=[anomalies.to_html(classes='data')], titles=anomalies.columns.values)

def detect_anomalies(df):
    model = IsolationForest(contamination=0.1)
    df['anomaly'] = model.fit_predict(df.select_dtypes(include=[float, int]))
    return df[df['anomaly'] == -1]

if __name__ == '__main__':
    app.run(debug=True)
