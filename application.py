from flask import Flask, render_template, request
from src.components.pipelines.predict_pipeline_file import CustomData, PredictPipeline


app = Flask(__name__)


@app.route('/DiamondPricePrediction', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    
    else:
        data = CustomData(
            color = request.form.get('color'),
            clarity = request.form.get('clarity'),
            depth = request.form.get('depth'),
            carat = request.form.get('carat'),
            table = request.form.get('table'),
            cut = request.form.get('cut'),
            x = request.form.get('x'),
            y = request.form.get('y'),
            z = request.form.get('z')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results = results[0])
    

if __name__ == '__main__':
    app.run('0.0.0.0', debug = True)