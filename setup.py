from setuptools import setup, find_packages

setup(
    name="telecom_churn_geo",
    version="1.0.0",
    description="Telecom Churn Prediction with Geospatial Segmentation",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "geopandas",
        "h3",
        "scikit-learn",
        "xgboost",
        "mlflow",
        "fastapi",
        "pydantic",
    ],
)
