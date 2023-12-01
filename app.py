from flask import Flask, render_template, request
from main import FuzzyController
app = Flask(__name__)

controller = FuzzyController()


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        motivation = request.form.get('personal_qualities', type=int)
        knowledge = request.form.get('knowledge', type=int)
        skills = request.form.get('skills', type=int)
        experience = request.form.get('experience', type=int)
        crisp = [motivation, knowledge, skills, experience]
        if min(crisp[:3]) < 1 or max(crisp[:3]) > 10 or experience < 0 or experience > 20:
            error = "Invalid Data!"
            return render_template('index.html', error=error)
        result = controller.get_result(crisp)
        print(result)
        return render_template('index.html', result=result)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)