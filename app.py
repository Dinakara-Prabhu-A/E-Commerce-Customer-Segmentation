import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from datetime import datetime
from src.components.text_processor import TextProcessor

def generate_recommendations(new_description):
    processor = TextProcessor()

    # Step 1: Load saved vectors, DataFrame, and vectorizer
    vectors, df, vectorizer = processor.load_data()

    # Step 2: Generate recommendations for the new description
    recommendations = processor.recommend_for_new_description(new_description, vectors, df, vectorizer, top_n=3)

    # Step 3: Return recommendations
    return recommendations


def predict_datapoint(main_container):
    """
    Predict the customer segment based on RFM inputs and provide product recommendations.
    """
    # Define the mapping of numeric results to text labels
    segment_mapping = {
        0: "Potential",
        1: "Frequent",
        2: "Loyal",
        4: "Inconsistent",
        5: "Dormant",
        6: "Bulk Purchase"
    }

    # Define the segment meanings
    segment_meanings = {
        "Potential": "You’re a new customer with great potential to shop more frequently.",
        "Frequent": "You shop regularly and are a valued customer.",
        "Loyal": "You are a dedicated customer, consistently making purchases.",
        "Inconsistent": "Your shopping habits are irregular, but targeted offers could engage you.",
        "Dormant": "You haven’t purchased recently, but we’re here to bring you back with exciting deals.",
        "Bulk Purchase": "You tend to buy in large quantities – let us offer you bulk discounts."
    }

    # Layout: Split into two columns
    col1, col2 = main_container.columns(2, gap='medium')

    # Column 1: Recency and Frequency Inputs
    with col1:
        
        container=st.container(border = True)
        container.markdown("#### Recency")
        recency = container.number_input(
                "🕒 How many days ago did you make your last purchase?",
                min_value=1, max_value=1000, value=1, step=1, key="recency1",
                help="Enter the number of days since your most recent purchase."
            )

        
        container=st.container(border = True)
        container.markdown("#### Frequency")
        frequency = container.number_input(
                "🔄 How many purchases have you made recently?",
                min_value=1, max_value=1000, value=1, step=1, key="frequency1",
                help="Enter the total number of purchases you've made in the given period."
            )
        
            

    # Column 2: Monetary and Description Inputs
    with col2:
        container=st.container(border = True)
        container.markdown("#### Monetary Value")
        monetary = container.number_input(
                "💰 What is the total amount you’ve spent recently?",
                min_value=1, max_value=100000, value=1, step=1, key="monetary1",
                help="Enter the total amount (in your currency) spent on purchases."
            )
        
        container=st.container(border = True)
        container.markdown("#### Product Description")
        new_description = container.text_input(
                "🛍️ Describe the product you’re looking for:",
                placeholder="e.g., A modern wooden chair with cushions.",
                help="Provide a brief description of the product to get recommendations."
            )   


    # Check if all inputs are filled
    if recency and frequency and monetary and new_description:

        # Prediction Button
        if st.button("✨ Predict Customer Segment", use_container_width=True):
            try:
                # Prepare data for prediction
                data = CustomData(recency=recency, frequency=frequency, monetary=monetary)
                pred_df = data.get_data_as_data_frame()
                
                # Run Prediction
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(pred_df)

                # Extract the result and map it to text
                result_value = int(''.join(filter(str.isdigit, str(results))))  # Extract integer result

                # Map to text based on the segment
                segment_text = segment_mapping.get(result_value, "Unknown Segment")

                # Display Predicted Customer Segment and Meaning with border
                st.markdown(
                    f"""
                    <div style="text-align: center; ">
                        <h2>🎯 Predicted Customer Segment:</h2>
                        <h1 style="color: skyblue; font-weight: bold;">{segment_text}</h1>
                        <hr style="border: 1px solid grey; width: 50%; margin: auto;">
                        <p style="font-size:18px;">🏷️ Based on your recent shopping activity, we’ve classified you as a <strong>{segment_text}</strong>. Here's what this means:</p>
                        <p style="font-size:18px;">{segment_meanings.get(segment_text, "We couldn’t determine your segment.")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Generate and Display Product Recommendations with border
                recommendations = generate_recommendations(new_description)
                st.markdown(
    f"""
    <div style="display: flex; justify-content: center; 
                align-items: flex-start; width: 50%; margin: 20px auto;">
        <div style="border: 2px solid green; 
                    border-radius: 20px; 
                    padding: 10px;">
            <h3>💡 Product Recommendations:</h3>
            <ul style="font-size:18px; text-align: left;">
                {''.join([f"<li style='margin-bottom: 10px;'>{rec}</li>" for rec in recommendations])}
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
                return segment_text
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(page_title='E-Commerce Customer Segmentation', 
                       page_icon=None, layout="wide", 
                       initial_sidebar_state="expanded", 
                       menu_items=None)
    st.markdown('''
        <style>
        [data-testid="stHeader"] {
        height: 0;
        
        }
        div.block-container {
        padding: 2rem;
    }
    </style>''',unsafe_allow_html=True
    )
    st.header('E-Commerce Customer Segmentation',divider = 'grey')
    st.markdown(
        """ <style>
            #e-commerence-customer-segementation {
            text-align: center;
            }
            </style>""",unsafe_allow_html=True
    )
    main_container = st.container(border = True)
    predict_datapoint(main_container)