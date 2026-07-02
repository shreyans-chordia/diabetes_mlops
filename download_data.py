import urllib.request

# Download the Pima Indians Diabetes dataset from GitHub
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
urllib.request.urlretrieve(url, "data/diabetes.csv")

# Add column headers to make the CSV self-documenting
headers = "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome"
with open("data/diabetes.csv", "r") as f:
    data = f.read()
with open("data/diabetes.csv", "w") as f:
    f.write(headers + "\n" + data)

print("Dataset downloaded: 768 rows, 9 columns")