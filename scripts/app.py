from flask import Flask, render_template, redirect, url_for, request, flash

app = Flask(__name__)

# Redirect root URL '/' to welcome page
@app.route('/')
def root():
    return redirect(url_for('welcome'))

# Add route for welcome page
@app.route('/welcome')
def welcome():
    from flask import flash
    flash("Test flash message: Welcome page loaded.", "info")
    return render_template('welcome.html')

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import predict as pc
import plant_disease

# Configuration
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id)) 

# Create DB
with app.app_context():
    db.create_all()

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'danger')
            return redirect(url_for('signup'))

        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Username already exists. Choose another.', 'danger')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(name=name, username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        db.session.refresh(new_user)  # Refresh to get updated user info
        flash('Signup successful! Welcome!', 'success')
        login_user(new_user)
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

# Home Route (Protected)
@app.route('/home')
@login_required
def home():
    title = 'Plant-Growth-Prediction'
    return render_template('index.html', title=title)
import os
# Growth Prediction Route
@app.route('/plant-growth-predict', methods=['GET', 'POST'])
@login_required
def plant_growth_prediction():
    title = 'Plant-Growth-Prediction'

    if request.method == 'POST':
        print("POST request received")
        if 'file' not in request.files:
            print("No file part in request.files")
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        print(f"File received: {file.filename}")

        if not file or file.filename == '':
            print("No file selected or filename empty")
            flash('No file selected', 'danger')
            return render_template('plant-growth.html', title=title)
        
        try:
            # Create 'input' directory if it doesn't exist
            input_folder = 'static/input'
            os.makedirs(input_folder, exist_ok=True)
            file_path = os.path.join(input_folder, file.filename)
            file.save(file_path)
            print(f"File saved to {file_path}")
            try:
                closest_stage, stages_resutl, additional_info_dict = pc.main(input_folder + "\\" + file.filename)
                print(f"Prediction results: closest_stage={closest_stage}, stages_resutl={stages_resutl}, additional_info_dict={additional_info_dict}")
            except Exception as e:
                print(f"Error in prediction: {e}")
                if "Invalid Image" in str(e):
                    flash("Invalid Image... Please Upload Valid Plant Image for Prediction.", "danger")
                else:
                    flash(f'Error: {str(e)}', 'danger')
                return render_template('plant-growth.html', title=title)
           
            # Ensure closest_stage is int and keys in additional_info_dict are int
            if isinstance(closest_stage, str):
                try:
                    closest_stage = int(closest_stage)
                except:
                    closest_stage = 1
            additional_info_dict = {int(k): v for k, v in additional_info_dict.items()}
            
            return render_template('plant-growth-prediction.html', disease={"Predicted Growth Stage": closest_stage}, image_filename=file.filename, image_path=stages_resutl, additional_info_dict=additional_info_dict, closest_stage=closest_stage, title=title)
        except Exception as e:
            print(f"Error saving file or processing: {e}")
            flash(f'Error: {str(e)}', 'danger')
            return redirect(request.url)
    print("GET request received")
    return render_template('plant-growth.html', title=title)

import pickle
from flask import render_template, session

# Load the trained model and label encoder pickle files
model = pickle.load(open('classifier.pkl', 'rb'))
ferti_encoder = pickle.load(open('fertilizer.pkl', 'rb'))

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('login'))

@app.route('/fertilizer-recommendation', methods=['GET', 'POST'])
@login_required
def fertilizer_recommendation():
    if request.method == 'POST':
        # Get form inputs
        temp = request.form.get('temp')
        humi = request.form.get('humid')
        mois = request.form.get('mois')
        soil = request.form.get('soil')
        crop = request.form.get('crop')
        nitro = request.form.get('nitro')
        pota = request.form.get('pota')
        phosp = request.form.get('phos')

        # Validate inputs
        if None in (temp, humi, mois, soil, crop, nitro, pota, phosp) or not all(val.isdigit() for val in (temp, humi, mois, soil, crop, nitro, pota, phosp)):
            error_msg = 'Invalid input. Please provide numeric values for all fields.'
            return render_template('fertilizer.html', recommendation=None, error=error_msg,
                                   temp=temp, humid=humi, mois=mois, soil=soil, crop=crop,
                                   nitro=nitro, pota=pota, phos=phosp)

        # Prepare input for prediction
        input_features = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]

        # Predict fertilizer class
        pred_class = model.predict([input_features])[0]

        # Decode fertilizer class to name
        fertilizer_name = ferti_encoder.inverse_transform([pred_class])[0]

        # Redirect to fertilizer_output route with query parameters to show output
        from flask import redirect, url_for
        return redirect(url_for('fertilizer_output', recommendation=fertilizer_name,
                                temp=temp, humid=humi, mois=mois, soil=soil, crop=crop,
                                nitro=nitro, pota=pota, phos=phosp))
    else:
        # Clear recommendation and inputs on GET request
        session.pop('recommendation', None)
        return render_template('fertilizer.html', recommendation=None)

@app.route('/fertilizer-output')
@login_required
def fertilizer_output():
    recommendation = request.args.get('recommendation')
    temp = request.args.get('temp')
    humid = request.args.get('humid')
    mois = request.args.get('mois')
    soil = request.args.get('soil')
    crop = request.args.get('crop')
    nitro = request.args.get('nitro')
    pota = request.args.get('pota')
    phos = request.args.get('phos')
    additional_info = request.args.get('additional_info')
    return render_template('fertilizer_output.html', recommendation=recommendation,
                           temp=temp, humid=humid, mois=mois, soil=soil, crop=crop,
                           nitro=nitro, pota=pota, phos=phos, additional_info=additional_info)

# Plant Disease Detection Routes
@app.route('/plant-disease', methods=['GET', 'POST'])
@login_required
def plant_disease_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        if file:
            input_folder = 'static/input'
            os.makedirs(input_folder, exist_ok=True)
            file_path = os.path.join(input_folder, file.filename)
            file.save(file_path)
            try:
                label, disease_info = plant_disease.predict_disease(file_path)
            except Exception as e:
                flash(f'Error in prediction: {str(e)}', 'danger')
                return redirect(request.url)
            return render_template('plant_disease_result.html', label=label, disease_info=disease_info, image_filename=file.filename)
    return render_template('plant_disease.html')

# Run the App
if __name__ == '__main__':
    app.run(debug=True)
