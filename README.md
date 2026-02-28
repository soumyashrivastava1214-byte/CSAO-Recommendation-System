# Cart Super Add-On (CSAO) Recommendation System

This project demonstrates a Cart Super Add-On (CSAO) recommendation system
that suggests relevant add-on items to users based on their current cart,
contextual signals, and learned behavior patterns.

The system is designed to mimic real-world food delivery platforms by
dynamically updating recommendations as the cart evolves.

---

## What this project does

- Builds a **session-based, context-aware recommendation system**
- Ranks candidate add-on items using a trained ML model
- Updates recommendations in real time as items are added to the cart
- Focuses on improving cart value and user experience

---

## How it works (high level)

1. User adds items to a cart
2. The system constructs the current cart context (cart size, value, last item, time context)
3. A trained model predicts the likelihood of adding each candidate add-on
4. Top-ranked add-ons are shown to the user (scores are hidden for UX clarity)

---

## Tech Stack

- Python
- XGBoost (for ranking add-on items)
- Pandas & Scikit-learn
- Streamlit (for interactive UI)

---

## Deployment

- The application is deployed using **Streamlit Cloud**
- The model is loaded from a pre-trained artifact
- Dataset is used as a mock catalog and feature store for demonstration

---

## Notes

- Raw prediction scores are not shown in the UI to avoid confusing users
- The focus is on **ranking quality**, not absolute probability values
- This project is intended as a functional prototype, not a production system
