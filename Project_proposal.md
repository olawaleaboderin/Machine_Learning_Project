# Project Title  
**Machine Learning for Genotype × Environment Interactions in Nigerian Maize Breeding Programs**

## Project Category  
**Tabular Data / Machine Learning Regression**

## Team Members  

| Name | Student ID |
|------|------------|
| Olawale Serifdeen Aboderin | 29206 |
| Francis Chinaecherem Uzor | 29260 |

---

## Project Description  

A major challenge in maize improvement is multilocational testing, which involves the evaluation of candidate varieties across diverse environments before official variety release. In Nigeria, maize breeders routinely evaluate hundreds of genotypes across multiple locations under three management conditions: drought, low-nitrogen, and optimum growing conditions, to assess performance, adaptability, and yield stability across target production environments prior to release. These trials are expensive, labor-intensive, and time-consuming.

The aim of this project is to apply machine learning to model Genotype × Environment (G×E) interactions in Nigerian maize breeding trials. The objectives are: (1) to predict grain yield of maize genotypes in untested environment-condition combinations, and (2) to identify trial locations that carry redundant environmental information. The project will deliver predictive models and a prototype Streamlit application for interactive visualization and yield prediction, supporting faster and more cost-effective maize variety development in Nigeria.

---

## Challenges  

- **G×E complexity:** Maize yield depends on the interaction between genotype and environment rather than either factor alone. Capturing these non-linear interactions is a key challenge for predictive modeling.  
- **High-dimensional feature space with mixed data types:** The dataset includes environmental variables, agronomic traits, and categorical identifiers alongside numerical measurements. Proper preprocessing and feature engineering are required to integrate these mixed data types and avoid overfitting.  
- **Multi-institution heterogeneity:** Data comes from different breeding institutions, which may differ in experimental design and germplasm. This introduces variability that the model must handle to ensure robust and generalizable predictions.  

---

## Dataset  

The dataset contains approximately 21,330 observations collected between 2020 and 2022 from five Nigerian locations under optimum, drought, and low-nitrogen conditions. It includes 237 maize genotypes from three breeding institutions and contains agronomic, environmental, and management-related variables, with grain yield as the target variable.

---

## Method and Algorithm  

Several machine learning models will be implemented and compared, including Ridge Regression, Lasso Regression, Support Vector Regression (SVR), Random Forest, and XGBoost. A classical AMMI (Additive Main Effects and Multiplicative Interaction) model will serve as the baseline. Data preprocessing and feature engineering will be performed using Python, Scikit-learn, and XGBoost.

---

## Evaluation  

Model performance will be evaluated using Root Mean Square Error (RMSE), Pearson correlation coefficient (r), and coefficient of determination (R²). Different cross-validation strategies will be used to simulate practical breeding scenarios, such as predicting performance in new environments or for previously untested genotypes.  

To optimize the trial network, clustering techniques such as K-Means and hierarchical clustering will be applied to identify environmentally similar locations. The results will help determine whether some testing sites can be removed without substantially reducing predictive accuracy, potentially lowering breeding costs while maintaining decision quality.
