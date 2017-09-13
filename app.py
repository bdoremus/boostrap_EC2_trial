from flask import Flask, request, render_template
import pickle as pickle
from build_model import TextClassifier
app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')+"""<h1>Welcome!</h1><br>
           <form action="/submit">
           <input type="submit" value = "Go to submit"/>
           </form>
           """

# Form page to submit text
@app.route('/submit')
def submission_page():
    # in this form, method = 'POST' means that data entered into
    # the 'user_input' variable will be sent to the /predict routing
    # block when the 'Enter text' submit button is pressed
    return '''
        <form action="/predict" method='POST' >
            <input type="text" name="words" />
            <input type="submit" value = 'Enter text'/>
        </form>
        '''

# Predict the group
# Recieves data in variable 'words' from the form in /submit
# and then processes it on the server and returns the category to the browser
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    text = request.form['words']
    with open('static/model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict([text])

    # formatted output string
    page = 'The predicted category is {0}.<br><br>The words used were:<br> {1}'

    # make html that gives us a button to go back to the home page
    resubmit_html = '''
        <form action="/submit" >
            <input type="submit" value = "Try Again"/>
        </form>
    '''

    return page.format(str(prediction), text) + resubmit_html


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
