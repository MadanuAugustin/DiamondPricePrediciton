from flask import Flask, render_template, request



app = Flask(__name__)


@app.route('/predictdata', methods = [GET, POST])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    
    else:
        data = CustomData(
            carat = request.form.get('carat'),
            color = request.form.get('color'),
            clarity = request.form.get('clarity'),
            depth = request.form.get('depth'),
            table = request.form.get('table'),
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