from flask import Flask, request, render_template
from flask_cors import CORS,cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


# Route for a home page

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            age=int(request.form.get('age')),
            sex=request.form.get('sex'),
            bmi=float(request.form.get('bmi')),
            children=int(request.form.get('children')),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('index.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    app.run(host="0.0.0.0", port=80)  # For Azure Cloud
