# TNEA Counsellor

A Flask web application that helps Tamil Nadu engineering aspirants explore colleges, predict admission chances, and get guidance for TNEA counselling.

## Features

- College recommendation based on cutoff, community, and preferred branches
- Dream, ambitious, and safe college categories
- TNEA cutoff data from 2022, 2023, and 2024
- College search with autocomplete suggestions
- AI chatbot for TNEA counselling questions
- Detailed AI-generated college information

## Tech Stack

- Python
- Flask
- Pandas
- Scikit-learn
- Random Forest Regressor
- HTML, CSS, and JavaScript
- OpenRouter API

## Project Structure

```text
files7/
├── app.py
├── model_backend.py
├── requirements.txt
├── dataset/
│   ├── tnea2022.csv
│   ├── tnea2023.csv
│   └── tnea2024.csv
└── templates/
    ├── index.html
    ├── predictor.html
    ├── chatbot.html
    └── college_search.html
