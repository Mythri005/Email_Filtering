# app.py
import streamlit as st
import pickle

# ----------------------------------------------------
# Load model and vectorizer
# ----------------------------------------------------
model = pickle.load(open('email_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ----------------------------------------------------
# Enhanced NLP Rule-based Refinement Function
# ----------------------------------------------------
def refine_category(email_text, base_pred):
    text = email_text.lower()

    # 🚫 Strong spam indicators
    spam_keywords = [
        "win", "claim", "prize", "lottery", "urgent action", "click here",
        "password", "bank account", "verify", "congratulations", "transfer",
        "money", "credit card", "guaranteed", "investment", "get rich",
        "cash reward", "limited offer now", "urgent response", "act now", "free iphone"
    ]

    # 🎯 Promotions indicators
    promo_keywords = [
        "offer", "discount", "sale", "coupon", "deal", "subscribe", "promo",
        "limited time", "save", "buy now", "special", "exclusive", "free gift",
        "new collection", "clearance", "voucher", "shopping", "festival sale"
    ]

    # 💼 Important/work-related indicators
    important_keywords = [
        "meeting", "project", "deadline", "report", "appointment",
        "schedule", "team", "presentation", "review", "exam", "submission"
    ]

    # 📦 Inbox/order-related indicators
    inbox_keywords = [
        "order", "invoice", "payment", "delivery", "tracking",
        "amazon", "flipkart", "shipped", "package", "receipt",
        "subscription", "newsletter"
    ]

    # ✅ Priority-based logic
    if any(word in text for word in spam_keywords):
        return "Spam"
    elif any(word in text for word in promo_keywords):
        # Check if also contains spam-like "win" or "free"
        if "win" in text or "free" in text:
            return "Spam"
        return "Promotions"
    elif any(word in text for word in important_keywords):
        return "Important"
    elif any(word in text for word in inbox_keywords):
        return "Inbox"
    else:
        return base_pred.capitalize()

# ----------------------------------------------------
# Streamlit App UI
# ----------------------------------------------------
st.set_page_config(page_title="📧 Smart Email Filter", layout="centered")

st.title("📧 Smart Email Filter using NLP")
st.write("This AI-powered app classifies your emails into **Spam**, **Important**, **Promotions**, or **Inbox** categories.")

email_text = st.text_area("✉️ Enter Email Text Below:")

if st.button("Classify Email"):
    if email_text.strip() == "":
        st.warning("Please enter an email text to classify.")
    else:
        # Vectorize input
        text_vector = vectorizer.transform([email_text])
        base_pred = model.predict(text_vector)[0]

        # Refine prediction
        final_category = refine_category(email_text, base_pred)

        # ----------------------------------------------------
        # Result Display with Colors
        # ----------------------------------------------------
        category_colors = {
            "Spam": "#ff4b4b",         # red
            "Promotions": "#ffb74d",   # orange
            "Important": "#64b5f6",    # blue
            "Inbox": "#81c784"         # green
        }

        color = category_colors.get(final_category, "#9e9e9e")

        st.markdown(
            f"<h3 style='text-align:center;'>Predicted Category: "
            f"<span style='color:{color}; font-weight:700;'>{final_category}</span></h3>",
            unsafe_allow_html=True
        )

        st.caption("Note: This classification uses both machine learning and keyword intelligence.")

# Footer
st.markdown("---")
st.caption("NLP-based Email Filtering")
