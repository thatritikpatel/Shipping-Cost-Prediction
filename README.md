# Shipping Cost Prediction

This is a **Machine Learning** project that predicts the cost of shipment based on various features such as weight, destination, and shipment type. The project utilizes a full pipeline starting from data exploration, feature engineering, model training, to model deployment using **FastAPI** and a **Web UI** for user interaction.

## Video Demo

Here's a video demo of the project:

https://github.com/user-attachments/assets/388f909e-18aa-4ab5-81f4-4c3e8f141bc2

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Project Components](#project-components)
    - [Data Preprocessing and Model Training](#data-preprocessing-and-model-training)
    - [Storing Data in MongoDB](#storing-data-in-mongodb)
    - [FastAPI Backend](#fastapi-backend)
    - [Frontend UI](#frontend-ui)
5. [Dependencies](#dependencies)
6. [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Running the FastAPI Server](#running-the-fastapi-server)
    - [Accessing the UI](#accessing-the-ui)
7. [Why ML Models Are Superior to Traditional Calculators](#why-ml-models-are-superior-to-traditional-calculators)
8. [Challenges](#challenges)
9. [Benefits](#benefits)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Overview

The **Shipping Cost Prediction** project is designed to predict the cost of shipment based on various input features. It incorporates a **machine learning model**, **data preprocessing**, and **a FastAPI backend** to deliver predictions via a simple web interface. The solution supports easy data storage in **MongoDB** for persistence and offers a clean web-based UI to make predictions.

- **ML Models**: XGBoost, CatBoost
- **Data Storage**: MongoDB
- **API**: FastAPI for backend server
- **Frontend UI**: HTML, CSS, and JavaScript for the user interface

---

## Folder Structure

The project follows a well-organized folder structure to keep each component modular. Here's how the folder structure is laid out:

```
Shipping-Cost-Prediction/
│
├── data/                     # Contains raw and processed datasets
│   ├── raw/                  # Raw data before any preprocessing
│   └── processed/            # Cleaned and processed data
│
├── notebooks/                # Jupyter notebooks for EDA and model training
│   ├── 1._EDA_Shipment-Pricing-Prediction.ipynb  # EDA (Exploratory Data Analysis)
│   └── 2._Feature_Engineering_Model_Shipment-Pricing-Prediction.ipynb  # Feature Engineering and Model Training
│
├── models/                   # Contains the trained machine learning models
│   ├── model.pkl             # Trained model
│   └── model_performance/    # Model evaluation and performance metrics
│
├── api/                      # FastAPI backend for predictions
│   ├── main.py               # FastAPI server script
│   └── prediction.py         # Logic for predictions
│
├── ui/                       # Frontend (HTML, CSS, JavaScript)
│   ├── index.html            # Main page for user input
│   ├── style.css             # Styles for the web UI
│   └── script.js             # JavaScript for handling form submission and displaying predictions
│
├── .env                      # Environment variables (e.g., MongoDB URI)
├── requirements.txt          # List of Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # Project license (MIT)
```

## Flow Charts

![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/flowchart/Data%20Ingestion%20Component.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/flowchart/Data%20Transformation%20Component.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/flowchart/Data%20Validation%20Component.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/flowchart/Model%20Evaluation%20Component.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/flowchart/Model%20Pusher%20Component.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/flowchart/Model%20Trainer%20Component.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/flowchart/model%20predictor.png)

## Code Flow

![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/codeflow/Data%20Ingestion%20Code%20Flow_2.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/codeflow/DataTransformation%20Code%20Flow_2.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/codeflow/DataValidation%20Code%20Flow_2.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/codeflow/Model%20Evaluation%20Code%20Flow_2.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/codeflow/Model%20predictor%20Code%20Flow.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/codeflow/ModelPusher%20Code%20Flow_2.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/codeflow/ModelTrainer%20Code%20Flow.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/codeflow/Training%20pipeline%20cofde%20flow_2.png)
![](https://github.com/thatritikpatel/Shipping-Cost-Prediction/blob/main/videos/Shipping%20Cost%20Prediction.mp4)

---

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

Ensure you have the following installed on your machine:

- **Python 3.8+**
- **MongoDB** (local or remote)

### Install Dependencies

Clone the repository and install the required Python dependencies:

```bash
git clone https://github.com/your-username/shipping-cost-prediction.git
cd shipping-cost-prediction
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file to store environment variables like the MongoDB connection string. Example `.env` file:

```env
MONGO_URI=mongodb://localhost:27017
```

---

## Project Components

### Data Preprocessing and Model Training

1. **Exploratory Data Analysis (EDA)**: 
   The file `1._EDA_Shipment-Pricing-Prediction.ipynb` contains the code for data exploration. In this step, we analyze the dataset, check for missing values, visualize distributions, and understand relationships between features and the target variable.

2. **Feature Engineering and Model Training**: 
   The file `2._Feature_Engineering_Model_Shipment-Pricing-Prediction.ipynb` covers feature engineering (like encoding categorical features and scaling numeric values) and model training. We train machine learning models (XGBoost and CatBoost) to predict shipping costs.

3. **Model Evaluation**: 
   We evaluate the trained models based on metrics such as RMSE and MAE, selecting the best-performing model for predictions.

### Storing Data in MongoDB

Data is stored in MongoDB for easy retrieval and updates. This includes storing the processed data, model results, and prediction outputs.

- The processed data is stored in the `data/processed/` directory.
- MongoDB stores the model, logs, and prediction results, accessible via the API.

### FastAPI Backend

The FastAPI app serves as the backend to handle prediction requests. The backend exposes an API endpoint that accepts shipment details, loads the model, and returns the predicted shipping cost.

- **`main.py`**: FastAPI server script that handles incoming requests.
- **`prediction.py`**: Logic to load the trained model and perform predictions.

### Frontend UI

The user interface (UI) is a simple web page that allows users to enter shipment details (like weight, destination, type, etc.) and get the predicted shipping cost.

- **`index.html`**: The main webpage where the user enters shipment data.
- **`style.css`**: The CSS file that provides styles for the UI.
- **`script.js`**: The JavaScript file that processes the form and sends requests to the backend API.

---

## Dependencies

Here’s a list of the required Python dependencies for the project:

```txt
annotated-types==0.7.0
boto3==1.35.16
category-encoders==2.6.3
cryptography==43.0.1
deprecation==2.1.0
dill==0.3.8
dynaconf==3.2.6
evidently==0.2.8
fastapi==0.114.0
from-root==1.3.0
fsspec==2024.9.0
httptools==0.6.1
iterative-telemetry==0.0.8
jaraco.text==3.12.1
jinja2==3.1.4
litestar==2.11.0
mypy-boto3-s3==1.35.16
ordered-set==4.1.0
pip-chill==1.0.3
platformdirs==4.2.2
pyarrow==17.0.0
pydantic-core==2.23.3
pymongo==4.8.0
python-dotenv==1.0.1
python-multipart==0.0.9
semantic-version==2.10.0
tomli==2.0.1
typer==0.12.5
typing-inspect==0.9.0
ujson==5.10.0
uvicorn==0.30.6
watchdog==4.0.2
watchfiles==0.24.0
websockets==13.0.1
xgboost==2.1.1
pandas
seaborn
ipykernel
matplotlib
catboost
scikit-learn
```

---

## Usage

### Training the Model

1. **Run the EDA Notebook**: Start by running `1._EDA_Shipment-Pricing-Prediction.ipynb` to explore the dataset, visualize relationships, and identify important features.
   
2. **Feature Engineering and Model Training**: Next, run `2._Feature_Engineering_Model_Shipment-Pricing-Prediction.ipynb`. This will process the data, engineer the necessary features, and train the machine learning models.

3. **Save the Model**: Once the best-performing model is selected, it is saved as `model.pkl` inside the `models/` folder.

### Running the FastAPI Server

After training the model, run the FastAPI server with the following command:

```bash
uvicorn api.main:app --reload
```

This starts the FastAPI server on `http://127.0.0.1:8000`, and the model is now ready to serve predictions.

### Accessing the UI

To access the UI, open the `ui/index.html` file in a browser. This will present a form where users can input shipment data. When the form is submitted, the backend API will respond with the predicted shipping cost.

---

## Why ML Models Are Superior to Traditional Calculators

Building a **shipment price predictor machine learning (ML) model** can provide substantial advantages over traditional calculators, even when the latter are accurate. Below are some key points explaining why ML models are more advantageous in dynamic environments:

### 1. **Dynamic and Complex Pricing**
   - **Dynamic Variables**: Shipment costs depend on fluctuating factors like fuel prices, seasonal demand, weather, and real-time availability of vehicles.
   - **Complex Interactions**: ML models can capture intricate, non-linear dependencies between factors such as delivery time, distance, and package size, which traditional calculators may fail to do.
   - **Advanced Features**: ML models can also consider customer preferences, geo-specific challenges, and historical demand trends.

### 2. **Customization for Business Use Cases**
   - ML models can incorporate business-specific pricing strategies, such as discounts, surcharges, and loyalty programs, making them more adaptable than static calculators.
   - They can also optimize pricing for internal operational costs and customer segmentation, allowing businesses to tailor pricing according to their needs.

### 3. **Scalability and Automation**
   - ML models can handle pricing for large-scale shipments without manual intervention, reducing human error and improving operational efficiency.

### 4. **Real-Time Adaptability**
   - ML models can adjust pricing based on real-time changes in market conditions, making them more responsive than static calculators.

### 5. **Enhanced Insights and Analytics**
   - ML models provide deeper insights into shipping trends, inefficiencies, and cost-saving opportunities, enabling data-driven decisions.

### 6. **Improved Customer Experience**
   - ML models can generate instantaneous quotes and suggest optimal shipping options based on price and speed, enhancing customer satisfaction.

### 7. **Anomaly Detection**
   - ML models can detect outliers and flag unusual prices, improving accuracy and reducing errors.

### 8. **Adaptability to Changing Conditions**
   - ML models can quickly adapt to changing market trends and regulatory shifts, ensuring that pricing remains competitive and compliant.

### 9. **Cost Efficiency and Profit Maximization**
   - By optimizing pricing, ML models help businesses maximize profits while maintaining competitive pricing structures.

### 10. **Competitive Differentiation**
   - Companies using ML models can differentiate themselves from competitors using static calculators by offering more accurate and adaptive pricing strategies.

---

## Challenges

While implementing an ML-based shipment cost predictor provides many benefits, there are also several challenges that come with it:

### 1. **Data Quality and Availability**
   - **Challenge**: Machine learning models depend heavily on high-quality data. Inconsistent or incomplete data can lead to poor model performance.
   - **Solution**: Ensuring robust data cleaning, feature selection, and handling of missing values is essential for accurate predictions.

### 2. **Model Complexity**
   - **Challenge**: ML models can become complex, requiring careful tuning and testing to achieve optimal performance. Hyperparameter optimization is a resource-intensive process.
   - **Solution**: Regular validation and testing, combined with techniques like cross-validation, can help manage model complexity and overfitting.

### 3. **Integration with Existing

 Systems**
   - **Challenge**: Integrating ML models into existing infrastructure (like legacy databases and software) can be a challenging task.
   - **Solution**: Modular design and API-based communication between the ML model and other components (like the frontend and database) can ease integration.

### 4. **Real-Time Prediction and Scaling**
   - **Challenge**: Predicting shipment costs in real time across a wide range of inputs can strain system resources.
   - **Solution**: Optimizing the model for inference speed and scaling the backend with appropriate infrastructure (e.g., cloud services) can ensure smooth real-time operation.

### 5. **User Acceptance**
   - **Challenge**: Users might prefer traditional methods and be skeptical about adopting ML-powered systems.
   - **Solution**: Demonstrating the accuracy and efficiency of the ML model through real-world examples and gradual transition can help in overcoming user resistance.

---

## Benefits

Despite the challenges, the use of machine learning for shipment cost prediction offers several compelling advantages:

### 1. **Accuracy and Precision**
   - ML models can predict shipping costs with higher accuracy, taking into account more variables than traditional calculators.

### 2. **Adaptability**
   - The ability to continuously retrain models with new data allows them to adapt to changes in market conditions, customer behavior, and other external factors.

### 3. **Efficiency**
   - Automation of the pricing process allows businesses to scale operations and handle a higher volume of shipments without manual intervention.

### 4. **Cost Savings**
   - By optimizing shipping routes, times, and methods, businesses can save on operational costs, which would otherwise increase due to inefficient pricing methods.

### 5. **Data-Driven Insights**
   - ML models can uncover trends and insights about customer preferences, seasonality, and demand fluctuations, providing valuable inputs for business strategies.

---

## Contributing

We welcome contributions! If you have suggestions or bug fixes, feel free to fork the repository, make changes, and submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details."# Shipping-Cost-Prediction" 

## Contact
- Ritik Patel - [https://www.linkedin.com/in/thatritikpatel/]
- Project Link: [https://github.com/thatritikpatel/Shipping-Cost-Prediction]
