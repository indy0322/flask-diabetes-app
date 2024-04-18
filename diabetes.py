import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import googleapiclient.discovery
import os
from flask import Flask, render_template
#from dotenv import load_dotenv

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF Score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')
    
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = np.array([[float(form.preg.data),
                           float(form.glucose.data),
                           float(form.blood.data),
                           float(form.skin.data),
                           float(form.insulin.data),
                           float(form.bmi.data),
                           float(form.dpf.data),
                           float(form.age.data)]])
        print(X_test.shape)
        print(X_test)
        
        data = pd.read_csv('./diabetes.csv', sep=',')
        
        X = data.values[:, 0:8]
        y = data.values[:, 8]
        
        scalar = MinMaxScaler()
        scalar.fit(X)
        
        X_test = scalar.transform(X_test)
        
        project_id = "ai-project-420400"
        model_id = "my_pima_model"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ai-project-420400-2a30b31fa6c3.json"
        model_path = "projects/{}/models/{}".format(project_id, model_id)
        model_path += "/versions/v0001/" 
        ml_resource = googleapiclient.discovery.build("ml", "v1").projects()
        
        input_data_json = {"signature_name": "serving_default","instances": X_test.tolist()}
        
        request = ml_resource.predict(name=model_path, body=input_data_json)
        response = request.execute()
        print("\nresponse: \n",response)
        
        if "error" in response:
            raise RuntimeError(response["error"])
        
        predD = np.array([pred['dense_2'] for pred in response["predictions"]])
        
        print(predD[0][0])
        res = predD[0][0]
        res = np.round(res, 2)
        res = (float)(np.round(res * 100))
        
        return render_template('result.html', res = res)
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()