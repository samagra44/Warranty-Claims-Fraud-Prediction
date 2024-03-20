# Warranty Claims Fraud Prediction

This project focuses on predicting warranty claims fraud using machine learning techniques. The dataset contains various features related to customer information, product details, service center details, and claim information. The goal is to build predictive models that can accurately identify fraudulent warranty claims based on the provided features.

## Dataset Details

The dataset comprises the following features:

- **Region**: Customer region details.
- **State**: Current location of the customer.
- **Area**: Urban/rural classification of the customer's area.
- **City**: Customer's current located city.
- **Consumer_profile**: Customer's work profile.
- **Product_category**: Category of the product.
- **Product_type**: Type of the product (e.g., TV, AC).
- **AC_1001_Issue**: Indicates failure of Compressor in AC (binary).
- **AC_1002_Issue**: Indicates failure of Condenser Coil in AC (binary).
- **AC_1003_Issue**: Indicates failure of Evaporator Coil in AC (binary).
- **TV_2001_Issue**: Indicates failure of power supply in TV (binary).
- **TV_2002_Issue**: Indicates failure of Inverter in TV (binary).
- **TV_2003_Issue**: Indicates failure of Motherboard in TV (binary).
- **Claim_value**: Customer's claim amount in Rs.
- **Service_Centre**: Service center details.
- **Product_Age**: Duration of the product purchased by the customer.
- **Purchased_from**: Source from where the product is purchased.
- **Call_details**: Call duration in minutes.
- **Purpose**: Purpose of the claim (compliant/claimed/other).
- **Fraud**: Target variable indicating fraudulent claim (1) or genuine claim (0).

## Models and Accuracy

The project utilizes various machine learning models to predict warranty claims fraud. The models and their corresponding accuracies are as follows:

- **DecisionTreeClassifier**: Accuracy = 0.918
- **RandomForestClassifier**: Accuracy = 0.916
- **LogisticRegression**: Accuracy = 0.913
- **SVC**: Accuracy = 0.911

## Repository Structure

```
├── data/
│   └── df_Clean.csv      # Dataset file
├── notebooks/
│   └── Warranty_Claims_Fraud_Detection.ipynb # Jupyter notebook with data exploration and model building
├── models/
│   └── decision_tree_model.pkl          # Pickled DecisionTreeClassifier model
│   └── random_forest_model.pkl          # Pickled RandomForestClassifier model
│   └── logistic_regression_model.pkl    # Pickled LogisticRegression model
│   └── svc_model.pkl                    # Pickled SVC model
├── README.md                            # Project README file
└── requirements.txt                     # Python dependencies
```

## Usage

1. Clone this repository:

```
git clone https://github.com/samagra44/Warranty-Claims-Fraud-Prediction.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Run the Jupyter notebook `warranty_claims_prediction.ipynb` in the `notebooks` directory to explore the dataset, build and evaluate machine learning models.

4. Alternatively, use the pre-trained models in the `models` directory for prediction on new data.

5. Run the application:
```
streamlit run app.py
```
## Acknowledgments

- This dataset is sourced from Kaggle [Warranty Claims Dataset](https://www.kaggle.com/competitions/warranty-claims/data).